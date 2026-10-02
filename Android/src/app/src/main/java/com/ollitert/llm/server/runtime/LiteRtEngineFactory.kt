/*
 * Copyright 2025 Google LLC
 * Modifications Copyright 2025-2026 @NightMean (https://github.com/NightMean)
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

import android.content.Context
import android.os.Environment
import android.os.StatFs
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Capabilities
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.ToolProvider
import com.ollitert.llm.server.R
import com.ollitert.llm.server.common.cleanUpLiteRtErrorMessage
import com.ollitert.llm.server.common.ServerMetrics
import com.ollitert.llm.server.data.model.Accelerator
import com.ollitert.llm.server.data.prefs.ConfigKeys
import com.ollitert.llm.server.data.prefs.DEFAULT_MAX_TOKEN
import com.ollitert.llm.server.data.prefs.DEFAULT_IMPORTED_MAX_CONTEXT_TOKENS
import com.ollitert.llm.server.data.prefs.DEFAULT_TEMPERATURE
import com.ollitert.llm.server.data.prefs.DEFAULT_TOPK
import com.ollitert.llm.server.data.prefs.DEFAULT_TOPP
import com.ollitert.llm.server.data.prefs.DEFAULT_VISION_ACCELERATOR
import com.ollitert.llm.server.data.model.EventCategory
import com.ollitert.llm.server.data.prefs.LOG_ERROR_PREVIEW_LONG_CHARS
import com.ollitert.llm.server.data.prefs.LOG_ERROR_PREVIEW_SHORT_CHARS
import com.ollitert.llm.server.data.model.LogLevel
import com.ollitert.llm.server.data.prefs.MIN_STORAGE_FOR_MODEL_INIT_BYTES
import com.ollitert.llm.server.data.model.Model
import com.ollitert.llm.server.data.model.ModelCapability
import com.ollitert.llm.server.data.model.modelMaxContextTokens
import com.ollitert.llm.server.data.repository.RequestLogStore
import com.ollitert.llm.server.data.prefs.SAMPLER_SEED_CONFIG_KEY
import com.ollitert.llm.server.data.prefs.ServerPrefs
import com.ollitert.llm.server.data.prefs.bytesToMb
import com.ollitert.llm.server.data.prefs.configSpeculativeDecodingEnabled
import java.io.File
import kotlin.random.Random
import com.ollitert.llm.server.data.prefs.isVerboseDebugEnabled
private const val TAG = "OlliteRT.EngineFactory"

internal fun Map<String, Any>?.samplerSeedOrRandom(): Int =
  (this?.get(SAMPLER_SEED_CONFIG_KEY) as? Number)?.toInt()
    ?: Random.nextInt(1, Int.MAX_VALUE)

object LiteRtEngineFactory {

  private data class ExperimentalFlagState(
    val isSpeculativeDecodingEnabled: Boolean?,
    val isConversationConstrainedDecodingEnabled: Boolean,
  )

  /**
   * Serializes every mutation of the process-global [ExperimentalFlags] statics.
   *
   * The flags are JVM-wide statics in the LiteRT SDK, but engine init runs on the
   * model-loader thread while conversation creation runs under the service's
   * inference lock on the executor thread (and a vision/audio reinit can overlap
   * a warmup load). Without this mutex, two threads flipping
   * `enableSpeculativeDecoding` / `enableConversationConstrainedDecoding` around
   * concurrent create calls can hand one of them the wrong native configuration.
   */
  private val nativeFlagScope = RestoredGlobalStateScope<ExperimentalFlagState>()

  @OptIn(ExperimentalApi::class)
  private fun captureExperimentalFlags(): ExperimentalFlagState {
    return ExperimentalFlagState(
      isSpeculativeDecodingEnabled = ExperimentalFlags.enableSpeculativeDecoding,
      isConversationConstrainedDecodingEnabled =
        ExperimentalFlags.enableConversationConstrainedDecoding,
    )
  }

  @OptIn(ExperimentalApi::class)
  private fun restoreExperimentalFlags(state: ExperimentalFlagState) {
    ExperimentalFlags.enableSpeculativeDecoding = state.isSpeculativeDecodingEnabled
    ExperimentalFlags.enableConversationConstrainedDecoding =
      state.isConversationConstrainedDecodingEnabled
  }

  /** Runs one native benchmark under the same flag ownership as engine construction. */
  @OptIn(ExperimentalApi::class)
  internal fun <Value> withSpeculativeDecodingFlag(
    isEnabled: Boolean,
    action: () -> Value,
  ): Value = nativeFlagScope.run(
    capture = ::captureExperimentalFlags,
    restore = ::restoreExperimentalFlags,
  ) {
    ExperimentalFlags.enableSpeculativeDecoding = isEnabled
    action()
  }


  /** Validates model file integrity and preflight storage before native engine instantiation. */
  fun validateModelFileAndStorage(context: Context, model: Model): String? {
    val modelPath = model.getPath(context = context)
    val modelFile = File(modelPath)
    if (!modelFile.exists()) {
      return context.getString(R.string.error_model_file_not_found, modelFile.name)
    }
    if (modelFile.length() < 1024) {
      return context.getString(R.string.error_model_file_corrupted, modelFile.length(), modelFile.name)
    }
    if (model.totalBytes > 0 && modelFile.length() != model.totalBytes) {
      Log.e(TAG, "Model file size mismatch: on-disk=${modelFile.length()}, expected=${model.totalBytes}")
      return context.getString(
        R.string.error_model_file_size_mismatch,
        modelFile.length().bytesToMb().toString() + "MB",
        model.totalBytes.bytesToMb().toString() + "MB",
      )
    }

    try {
      val stat = StatFs(Environment.getDataDirectory().path)
      val availableBytes = stat.availableBlocksLong * stat.blockSizeLong
      if (availableBytes < MIN_STORAGE_FOR_MODEL_INIT_BYTES) {
        val availableMb = availableBytes.bytesToMb()
        val requiredMb = MIN_STORAGE_FOR_MODEL_INIT_BYTES.bytesToMb()
        return context.getString(R.string.error_storage_insufficient, availableMb.toString(), requiredMb.toString())
      }
    } catch (e: Exception) {
      Log.w(TAG, "Failed to check storage before engine creation: ${e.message}")
    }
    return null
  }

  data class BackendResolution(
    val preferredBackend: Backend,
    val visionBackend: Backend,
    val effectiveAccelerator: String,
    val canFallbackToCpu: Boolean,
  )

  fun resolveBackends(
    context: Context,
    model: Model,
    configOverrides: Map<String, Any>?,
  ): BackendResolution {
    val accelerator = configOverrides?.let {
      (it[ConfigKeys.ACCELERATOR.id] as? String) ?: Accelerator.GPU.label
    } ?: model.getStringConfigValue(key = ConfigKeys.ACCELERATOR, defaultValue = Accelerator.GPU.label)

    val visionAccelerator = configOverrides?.let {
      (it[ConfigKeys.VISION_ACCELERATOR.id] as? String) ?: DEFAULT_VISION_ACCELERATOR.label
    } ?: model.getStringConfigValue(
      key = ConfigKeys.VISION_ACCELERATOR,
      defaultValue = DEFAULT_VISION_ACCELERATOR.label,
    )

    val gpuAccessible = GpuAvailability.isOpenClAccessible
    val canFallbackToCpu = model.accelerators.contains(Accelerator.CPU)

    Log.i(TAG, "Backend selection: requested=$accelerator, openCL=$gpuAccessible, " +
      "cpuFallbackAvailable=$canFallbackToCpu, accelerators=${model.accelerators}")
    if (ServerPrefs.isVerboseDebugEnabled(context)) {
      RequestLogStore.addEvent(
        "Backend: requested=$accelerator, OpenCL=${if (gpuAccessible) "OK" else "unavailable"}, " +
          "accelerators=${model.accelerators.map { it.label }}",
        level = LogLevel.DEBUG,
        modelName = model.name,
        category = EventCategory.MODEL,
      )
    }

    val effectiveAccelerator = if (accelerator == Accelerator.GPU.label && !gpuAccessible && canFallbackToCpu) {
      Log.w(TAG, "GPU requested but OpenCL not accessible — falling back to CPU")
      RequestLogStore.addEvent(
        "GPU unavailable (OpenCL not accessible), using CPU",
        level = LogLevel.WARNING,
        modelName = model.name,
        category = EventCategory.MODEL,
      )
      Accelerator.CPU.label
    } else {
      accelerator
    }

    val effectiveVisionAccelerator = if (visionAccelerator == Accelerator.GPU.label && !gpuAccessible) {
      Accelerator.CPU.label
    } else {
      visionAccelerator
    }

    val visionBackend = when (effectiveVisionAccelerator) {
      Accelerator.CPU.label -> Backend.CPU()
      Accelerator.GPU.label -> Backend.GPU()
      Accelerator.NPU.label,
      Accelerator.TPU.label -> Backend.NPU(nativeLibraryDir = context.applicationInfo.nativeLibraryDir)
      else -> Backend.GPU()
    }

    val preferredBackend = when (effectiveAccelerator) {
      Accelerator.CPU.label -> Backend.CPU()
      Accelerator.GPU.label -> Backend.GPU()
      Accelerator.NPU.label,
      Accelerator.TPU.label -> Backend.NPU(nativeLibraryDir = context.applicationInfo.nativeLibraryDir)
      else -> Backend.CPU()
    }

    return BackendResolution(
      preferredBackend = preferredBackend,
      visionBackend = visionBackend,
      effectiveAccelerator = effectiveAccelerator,
      canFallbackToCpu = canFallbackToCpu,
    )
  }

  @OptIn(ExperimentalApi::class)
  fun createAndInitEngine(
    context: Context,
    model: Model,
    supportImage: Boolean,
    supportAudio: Boolean,
    configOverrides: Map<String, Any>?,
    systemInstruction: Contents?,
    tools: List<ToolProvider>,
    initialMessages: List<Message>,
    enableConversationConstrainedDecoding: Boolean,
    onDone: (String) -> Unit,
  ) {
    // A failed reload must not leave the previous engine capacity visible to the UI or context
    // manager. The native attempt below is transactional and will repopulate these fields only
    // after both engine and conversation creation succeed.
    model.engineContextTokens = 0
    model.engineContextSource = "unknown"
    model.engineMaxOutputTokens = 0
    model.engineSafetyReserveTokens = 0
    model.engineUsableInputTokens = 0
    val validationError = validateModelFileAndStorage(context, model)
    if (validationError != null) {
      onDone(validationError)
      return
    }

    val maxTokens = configOverrides?.let {
      (it[ConfigKeys.MAX_TOKENS.id] as? Number)?.toInt() ?: DEFAULT_MAX_TOKEN
    } ?: model.getIntConfigValue(key = ConfigKeys.MAX_TOKENS, defaultValue = DEFAULT_MAX_TOKEN)
    val topK = configOverrides?.let {
      (it[ConfigKeys.TOPK.id] as? Number)?.toInt() ?: DEFAULT_TOPK
    } ?: model.getIntConfigValue(key = ConfigKeys.TOPK, defaultValue = DEFAULT_TOPK)
    val topP = configOverrides?.let {
      (it[ConfigKeys.TOPP.id] as? Number)?.toFloat() ?: DEFAULT_TOPP
    } ?: model.getFloatConfigValue(key = ConfigKeys.TOPP, defaultValue = DEFAULT_TOPP)
    val temperature = configOverrides?.let {
      (it[ConfigKeys.TEMPERATURE.id] as? Number)?.toFloat() ?: DEFAULT_TEMPERATURE
    } ?: model.getFloatConfigValue(key = ConfigKeys.TEMPERATURE, defaultValue = DEFAULT_TEMPERATURE)
    val seed = configOverrides.samplerSeedOrRandom()

    val backendResolution = resolveBackends(context, model, configOverrides)
    val preferredBackend = backendResolution.preferredBackend
    val visionBackend = backendResolution.visionBackend
    val canFallbackToCpu = backendResolution.canFallbackToCpu
    val modelPath = model.getPath(context = context)

    val declaredMaxContextTokens = (configOverrides?.get(ConfigKeys.MAX_CONTEXT_LENGTH.id) as? Number)?.toInt()
      ?: model.modelMaxContextTokens
      ?: DEFAULT_IMPORTED_MAX_CONTEXT_TOKENS
    // The store is read here, at the orchestration boundary, because only the attempt loop below
    // knows which tier actually initialized. The planner itself stays pure.
    val tierStore = ContextCapacitySupport.tierStore()
    val capacityPlan = ContextCapacityPlanner.plan(
      modelMaxContextTokens = declaredMaxContextTokens,
      maxOutputTokens = maxTokens,
      backend = preferredBackend,
      memory = ContextCapacitySupport.currentMemory(),
      preferredTier = tierStore.readPreferredTier(
        modelId = model.name,
        modelVersion = model.version,
        backend = preferredBackend,
      ),
      modelSizeBytes = model.totalBytes,
      modelMinDeviceMemoryInGb = model.minDeviceMemoryInGb,
    )
    Log.i(TAG, "Context capacity for '${model.name}': modelMax=${capacityPlan.modelMaxContextTokens}, " +
      "engine=${capacityPlan.engineContextTokens}, usableInput=${capacityPlan.usableInputTokens}, " +
      "maxOutput=${capacityPlan.maxOutputTokens}, source=${capacityPlan.source}, " +
      "tiers=${capacityPlan.candidateTiers}")

    fun engineConfigFor(backend: Backend, vision: Backend, contextTokens: Int) = EngineConfig(
      modelPath = modelPath,
      backend = backend,
      visionBackend = if (supportImage) vision else null,
      // GPU audio acceleration is opt-in (Settings → Model Behaviour): the
      // executor may be unsupported on some devices, so CPU stays the default.
      audioBackend = when {
        !supportAudio -> null
        ServerPrefs.isAudioGpuAccelerationEnabled(context) -> Backend.GPU()
        else -> Backend.CPU()
      },
      maxNumTokens = contextTokens,
      cacheDir =
        if (modelPath.startsWith("/data/local/tmp"))
          context.getExternalFilesDir(null)?.absolutePath
        else null,
    )

    var supportsSpeculativeDecoding = false
    try {
      Capabilities(modelPath).use {
        supportsSpeculativeDecoding = it.hasSpeculativeDecodingSupport()
      }
    } catch (e: Exception) {
      Log.w(TAG, "Capabilities probe failed for '${model.name}': ${e.message}")
    }

    val specDecUserEnabled = configOverrides?.configSpeculativeDecodingEnabled()
      ?: model.configValues.configSpeculativeDecodingEnabled()
      ?: false
    val enableSpeculativeDecoding = supportsSpeculativeDecoding &&
      ModelCapability.SPECULATIVE_DECODING in model.capabilities &&
      specDecUserEnabled

    val backendAttempts = buildList {
      add(preferredBackend to visionBackend)
      if (preferredBackend is Backend.GPU && canFallbackToCpu) {
        add(Backend.CPU() to Backend.CPU())
      }
    }
    var lastFailure: Throwable? = null
    var successfulBackend: Backend? = null
    var successfulContext = 0

    for ((backend, vision) in backendAttempts) {
      if (backend is Backend.CPU && preferredBackend is Backend.GPU) {
        Log.w(TAG, "GPU initialization failed or was unavailable, retrying with CPU backend")
        ServerMetrics.onCpuFallbackStarted()
        RequestLogStore.addEvent(
          "Retrying model initialization on CPU",
          level = LogLevel.WARNING,
          modelName = model.name,
          category = EventCategory.MODEL,
        )
      }
      for (contextTokens in capacityPlan.candidateTiers) {
        try {
          model.instance = initEngineWithConversation(
            engineConfig = engineConfigFor(backend, vision, contextTokens),
            enableSpeculativeDecoding = enableSpeculativeDecoding && backend !is Backend.CPU,
            enableConversationConstrainedDecoding = enableConversationConstrainedDecoding,
            topK = topK,
            topP = topP,
            temperature = temperature,
            seed = seed,
            systemInstruction = systemInstruction,
            tools = tools,
            initialMessages = initialMessages,
          )
          successfulBackend = backend
          successfulContext = contextTokens
          break
        } catch (attemptFailure: Throwable) {
          lastFailure = attemptFailure
          model.instance = null
          Log.w(TAG, "Engine init failed for '${model.name}' on " +
            "${backend::class.simpleName} context=$contextTokens: " +
            "[${attemptFailure::class.simpleName}] ${attemptFailure.message}")
          RequestLogStore.addEvent(
            "Engine init failed backend=${backend::class.simpleName} context=$contextTokens: " +
              attemptFailure.message?.take(LOG_ERROR_PREVIEW_SHORT_CHARS),
            level = LogLevel.WARNING,
            modelName = model.name,
            category = EventCategory.MODEL,
          )
        }
      }
      if (successfulBackend != null) break
    }

    val backend = successfulBackend
    if (backend == null || model.instance == null) {
      val failure = lastFailure
      RequestLogStore.addEvent(
        "All engine initialization attempts failed: [${failure?.javaClass?.simpleName}] " +
          failure?.message?.take(LOG_ERROR_PREVIEW_LONG_CHARS),
        level = LogLevel.ERROR,
        modelName = model.name,
        category = EventCategory.MODEL,
      )
      onDone(cleanUpLiteRtErrorMessage(failure?.message ?: context.getString(R.string.error_unknown)))
      return
    }

    model.engineContextTokens = successfulContext
    model.engineContextSource = if (successfulContext == capacityPlan.engineContextTokens) {
      capacityPlan.source
    } else {
      "native_fallback"
    }
    model.engineMaxOutputTokens = capacityPlan.maxOutputTokens.coerceAtMost(
      (successfulContext - capacityPlan.safetyReserveTokens - ContextCapacityPlanner.MIN_INPUT_TOKENS)
        .coerceAtLeast(1),
    )
    model.engineSafetyReserveTokens = capacityPlan.safetyReserveTokens
    model.engineUsableInputTokens = (successfulContext - model.engineMaxOutputTokens - model.engineSafetyReserveTokens)
      .coerceAtLeast(1)
    // Recorded only now, after the engine genuinely initialized on this backend.
    tierStore.recordSuccessfulTier(
      modelId = model.name,
      modelVersion = model.version,
      backend = backend,
      tier = successfulContext,
    )
    Log.i(TAG, "Engine initialized successfully on ${backend::class.simpleName} for '${model.name}' " +
      "context=$successfulContext source=${model.engineContextSource} " +
      "(speculative_decoding=${enableSpeculativeDecoding && backend !is Backend.CPU})")
    RequestLogStore.addEvent(
      "Engine initialized on ${backend::class.simpleName} context=$successfulContext " +
        if (enableSpeculativeDecoding && backend !is Backend.CPU) "(MTP enabled)" else "",
      level = LogLevel.INFO,
      modelName = model.name,
      category = EventCategory.MODEL,
    )
    onDone("")
  }

  /**
   * Builds an [Engine] from [engineConfig], initializes it, and creates the initial
   * [Conversation]. Shared by the preferred-backend path and the CPU fallback path in
   * [createAndInitEngine]; callers only decide which backend config to pass and how to
   * log/report outcomes.
   *
   * Owns resource safety for the half-constructed pair: if engine creation, initialization,
   * or conversation creation throws, the engine is closed and GC is hinted before rethrowing.
   */
  @OptIn(ExperimentalApi::class)
  private fun initEngineWithConversation(
    engineConfig: EngineConfig,
    enableSpeculativeDecoding: Boolean,
    enableConversationConstrainedDecoding: Boolean,
    topK: Int,
    topP: Float,
    temperature: Float,
    seed: Int,
    systemInstruction: Contents?,
    tools: List<ToolProvider>,
    initialMessages: List<Message>,
  ): LlmModelInstance = nativeFlagScope.run(
    capture = ::captureExperimentalFlags,
    restore = ::restoreExperimentalFlags,
  ) {
    ExperimentalFlags.enableSpeculativeDecoding = enableSpeculativeDecoding
    var engine: Engine? = null
    try {
      val initializedEngine = Engine(engineConfig)
      engine = initializedEngine
      initializedEngine.initialize()
      ExperimentalFlags.enableSpeculativeDecoding = false

      ExperimentalFlags.enableConversationConstrainedDecoding =
        enableConversationConstrainedDecoding
      // NPU backends reject custom sampler configs, so only non-NPU gets one.
      val useSampler = engineConfig.backend !is Backend.NPU
      val conversation = initializedEngine.createConversation(
        ConversationConfig(
          // Flag only gates whether per-message ResponseFormat is permitted —
          // it has no effect unless a request actually passes one. Always-on
          // so json_schema requests can use native constrained decoding.
          enableResponseFormat = true,
          samplerConfig =
            if (useSampler) {
              SamplerConfig(
                topK = topK,
                topP = topP.toDouble(),
                temperature = temperature.toDouble(),
                seed = seed,
              )
            } else {
              null
            },
          systemInstruction = systemInstruction,
          tools = tools,
          initialMessages = initialMessages,
          automaticToolCalling = false,
        ),
      )
      LlmModelInstance(engine = initializedEngine, conversation = conversation)
    } catch (failure: Throwable) {
      try { engine?.close() } catch (closeEx: Exception) {
        Log.w(TAG, "Engine.close() failed during init cleanup", closeEx)
      }
      System.gc()
      throw failure
    }
  }

  @OptIn(ExperimentalApi::class)
  fun createConversation(
    engine: Engine,
    model: Model,
    systemInstruction: Contents?,
    tools: List<ToolProvider>,
    initialMessages: List<Message>,
    enableConversationConstrainedDecoding: Boolean,
  ): Conversation {
    val topK = model.getIntConfigValue(key = ConfigKeys.TOPK, defaultValue = DEFAULT_TOPK)
    val topP = model.getFloatConfigValue(key = ConfigKeys.TOPP, defaultValue = DEFAULT_TOPP)
    val temperature =
      model.getFloatConfigValue(key = ConfigKeys.TEMPERATURE, defaultValue = DEFAULT_TEMPERATURE)
    val seed = model.configValues.samplerSeedOrRandom()

    val accelerator =
      model.getStringConfigValue(
        key = ConfigKeys.ACCELERATOR,
        defaultValue = Accelerator.GPU.label,
      )
    return nativeFlagScope.run(
      capture = ::captureExperimentalFlags,
      restore = ::restoreExperimentalFlags,
    ) {
      ExperimentalFlags.enableConversationConstrainedDecoding =
        enableConversationConstrainedDecoding
      val isNpuBackend = accelerator == Accelerator.NPU.label || accelerator == Accelerator.TPU.label
      engine.createConversation(
        ConversationConfig(
          // Flag only gates whether per-message ResponseFormat is permitted —
          // it has no effect unless a request actually passes one. Always-on
          // so json_schema requests can use native constrained decoding.
          enableResponseFormat = true,
          samplerConfig =
            if (!isNpuBackend) {
              SamplerConfig(
                topK = topK,
                topP = topP.toDouble(),
                temperature = temperature.toDouble(),
                seed = seed,
              )
            } else {
              null
            },
          systemInstruction = systemInstruction,
          tools = tools,
          initialMessages = initialMessages,
          automaticToolCalling = false,
        )
      )
    }
  }
}
