/*
 * Copyright 2025-2026 @NightMean (https://github.com/NightMean)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.ollitert.llm.server.runtime

import com.google.ai.edge.litertlm.Backend
import kotlin.math.max
import kotlin.math.min

/**
 * Device memory as this planner needs it: three plain numbers, owned by OlliteRT.
 *
 * Deliberately not a host type. A host snapshot type would reintroduce the host dependency
 * this planner was extracted to remove.
 */
data class ContextMemoryView(
  val totalBytes: Long,
  val availableBytes: Long,
  val lowMemory: Boolean,
)

/** Result of [ContextCapacityPlanner.plan]. Contains no host types. */
data class ContextCapacityPlan(
  val modelMaxContextTokens: Int,
  val engineContextTokens: Int,
  val maxOutputTokens: Int,
  val safetyReserveTokens: Int,
  val usableInputTokens: Int,
  val source: String,
  val candidateTiers: List<Int>,
)

/**
 * Selects a stable LiteRT-LM context allocation for one model load.
 *
 * Pure and deterministic: every input is a parameter and nothing is read from disk, prefs, or
 * ambient state. The caller owns remembering which tier worked and passes it back in as
 * [preferredTier] — persistence is an optimization, never a prerequisite for planning.
 *
 * [EngineConfig.maxNumTokens] is the combined input/output capacity and therefore controls the
 * native KV-cache allocation. This planner never treats the model's output ceiling as its context
 * capacity. Native initialization remains authoritative: callers try the returned tiers in order
 * and record the first successful one.
 */
object ContextCapacityPlanner {
  const val MIN_INPUT_TOKENS = 512
  private const val SAFETY_RESERVE_TOKENS = 1_000
  private const val BYTES_PER_GB = 1024L * 1024L * 1024L
  private const val LOW_MEMORY_BYTES = 768L * 1024L * 1024L
  private const val LIMITED_MEMORY_BYTES = 1_536L * 1024L * 1024L
  private const val MODEL_LOAD_RESERVE_CAP_BYTES = 2L * BYTES_PER_GB
  private const val MAX_PLANNED_CONTEXT_TOKENS = 32_768
  private const val GPU_MEMORY_RESERVE_BYTES = 256L * 1024L * 1024L
  private const val NPU_MEMORY_RESERVE_BYTES = 256L * 1024L * 1024L
  private const val CPU_MEMORY_RESERVE_BYTES = 128L * 1024L * 1024L

  /** Deterministic tiers make probing, diagnostics, and regression checks reproducible. */
  private val CONTEXT_TIERS = listOf(32_768, 32_000, 24_576, 16_384, 12_288, 8_192, 4_096)

  fun plan(
    modelMaxContextTokens: Int,
    maxOutputTokens: Int,
    backend: Backend,
    memory: ContextMemoryView?,
    preferredTier: Int?,
    modelSizeBytes: Long,
    modelMinDeviceMemoryInGb: Int?,
  ): ContextCapacityPlan {
    val modelMax = modelMaxContextTokens.coerceIn(1, MAX_PLANNED_CONTEXT_TOKENS)
    // Keep output and context as separate concepts, but never choose an output reserve that
    // makes the smallest valid engine allocation exceed the model's declared capacity.
    val output = maxOutputTokens.coerceIn(
      1,
      (modelMax - SAFETY_RESERVE_TOKENS - MIN_INPUT_TOKENS).coerceAtLeast(1),
    )
    val minimumEngineContext = output + SAFETY_RESERVE_TOKENS + MIN_INPUT_TOKENS
    val memoryCap = memory?.let {
      memoryContextCap(
        memory = it,
        modelSizeBytes = modelSizeBytes,
        modelMinDeviceMemoryInGb = modelMinDeviceMemoryInGb,
        modelMaxContextTokens = modelMax,
        backend = backend,
      )
    }
    val allowedMax = min(modelMax, memoryCap ?: modelMax)
    val tiers = CONTEXT_TIERS
      .filter { it <= allowedMax && it >= minimumEngineContext }
      .ifEmpty { listOf(allowedMax.coerceAtLeast(1)) }
      .distinct()
      .sortedDescending()

    // A remembered tier only counts when it is still a legal candidate under current conditions.
    val cached = preferredTier?.takeIf { it in tiers }
    val orderedTiers = orderCandidateTiers(
      tiers = tiers,
      cached = cached,
      allowUpwardRecovery = memoryCap == null || allowedMax > (cached ?: 0),
    )
    val selected = orderedTiers.first()
    val source = when {
      cached != null -> "cached"
      memory == null -> "native_probe"
      memoryCap != null && memoryCap < modelMax -> "memory_adaptive"
      else -> "memory_probe"
    }
    return ContextCapacityPlan(
      modelMaxContextTokens = modelMax,
      engineContextTokens = selected,
      maxOutputTokens = output,
      safetyReserveTokens = SAFETY_RESERVE_TOKENS,
      usableInputTokens = (selected - output - SAFETY_RESERVE_TOKENS).coerceAtLeast(1),
      source = source,
      candidateTiers = orderedTiers,
    )
  }

  /**
   * Orders a persisted tier as a hint, not a ceiling. When current memory permits a larger
   * tier, retry larger tiers first so a temporary low-memory downgrade can recover naturally.
   */
  fun orderCandidateTiers(
    tiers: List<Int>,
    cached: Int?,
    allowUpwardRecovery: Boolean,
  ): List<Int> {
    val descending = tiers.distinct().sortedDescending()
    if (cached == null || allowUpwardRecovery) return descending
    return listOf(cached) + descending.filter { it < cached }
  }

  private fun memoryContextCap(
    memory: ContextMemoryView,
    modelSizeBytes: Long,
    modelMinDeviceMemoryInGb: Int?,
    modelMaxContextTokens: Int,
    backend: Backend,
  ): Int? {
    if (memory.totalBytes <= 0L || memory.availableBytes <= 0L) return null
    if (memory.lowMemory || memory.availableBytes <= LOW_MEMORY_BYTES) return 8_192

    val modelMinBytes = modelMinDeviceMemoryInGb?.toLong()?.times(BYTES_PER_GB)
    if (modelMinBytes != null && memory.totalBytes < modelMinBytes) {
      return min(modelMaxContextTokens, 16_384)
    }

    // The model file is not assumed to equal resident native memory. Reserve a bounded,
    // conservative fraction of it so a pre-load probe does not treat all free RAM as KV-cache
    // headroom. Successful native initialization remains the final authority.
    val modelLoadReserve = min(
      MODEL_LOAD_RESERVE_CAP_BYTES,
      max(0L, modelSizeBytes / 2L),
    )
    val backendReserve = when (backend) {
      is Backend.GPU -> GPU_MEMORY_RESERVE_BYTES
      is Backend.NPU -> NPU_MEMORY_RESERVE_BYTES
      else -> CPU_MEMORY_RESERVE_BYTES
    }
    val estimatedHeadroom = memory.availableBytes - modelLoadReserve - backendReserve
    return when {
      estimatedHeadroom <= LOW_MEMORY_BYTES -> min(modelMaxContextTokens, 8_192)
      estimatedHeadroom <= LIMITED_MEMORY_BYTES -> min(modelMaxContextTokens, 16_384)
      else -> modelMaxContextTokens
    }
  }
}
