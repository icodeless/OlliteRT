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

/**
 * Remembers which context tier actually initialized successfully, so the next load can try it
 * first. Implementations own their storage, namespace, and lifetime.
 *
 * This is a store, not a cache: the remembered tier changes subsequent allocation strategy.
 * Callers must treat [readPreferredTier] as a hint that [ContextCapacityPlanner] is free to
 * reject, and must only call [recordSuccessfulTier] after engine creation has genuinely
 * succeeded — the store is never told about an attempted-but-failed tier.
 */
interface ContextCapacityTierStore {
  fun readPreferredTier(modelId: String, modelVersion: String, backend: Backend): Int?
  fun recordSuccessfulTier(modelId: String, modelVersion: String, backend: Backend, tier: Int)
}

/** Default for hosts that do not persist tier history. Planning then runs from native probes. */
object NoOpContextCapacityTierStore : ContextCapacityTierStore {
  override fun readPreferredTier(modelId: String, modelVersion: String, backend: Backend): Int? = null
  override fun recordSuccessfulTier(modelId: String, modelVersion: String, backend: Backend, tier: Int) = Unit
}

/**
 * Supplies current device memory to [ContextCapacityPlanner]. Returns null when memory data is
 * unavailable, which makes the planner fall back to native probing.
 */
fun interface ContextMemoryProvider {
  fun currentMemory(): ContextMemoryView?
}

/**
 * Injection point for host-supplied capacity collaborators.
 *
 * Defaults are deliberately empty so a standalone OlliteRT build compiles and runs with no host
 * present: no memory provider means native probing, and the no-op store means no tier history.
 * Neither degrades correctness — native initialization remains the final authority.
 */
object ContextCapacitySupport {
  @Volatile
  private var memoryProvider: ContextMemoryProvider? = null

  @Volatile
  private var tierStore: ContextCapacityTierStore = NoOpContextCapacityTierStore

  fun install(memoryProvider: ContextMemoryProvider?, tierStore: ContextCapacityTierStore) {
    this.memoryProvider = memoryProvider
    this.tierStore = tierStore
  }

  fun currentMemory(): ContextMemoryView? = try {
    memoryProvider?.currentMemory()
  } catch (_: Throwable) {
    // Memory telemetry is advisory; a failing provider must not block engine initialization.
    null
  }

  fun tierStore(): ContextCapacityTierStore = tierStore
}
