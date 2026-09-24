/*
 * Copyright 2025-2026 @NightMean (https://github.com/NightMean)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

package com.ollitert.llm.server.service.inference

import android.os.SystemClock
import kotlinx.serialization.Serializable
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

/** The terminal callback observed directly at the LiteRT-LM boundary. */
@Serializable
enum class NativeTerminalSignal {
  ON_DONE,
  CANCELLATION_EXCEPTION,
  ON_ERROR,
  UNAVAILABLE,
}

/** The terminal decision made by OlliteRT's request gateway. */
@Serializable
enum class GatewayTerminalOutcome {
  SUCCESS,
  TIMEOUT,
  CANCELLED,
  ERROR,
  STOPPED,
  UNAVAILABLE,
}

/** Which progress-aware watchdog boundary, if any, terminated the request. */
@Serializable
enum class TimeoutKind {
  NONE,
  STARTUP,
  IDLE,
}

/** Centralized progress-aware watchdog policy. There is intentionally no total-generation limit. */
data class InferenceWatchdogPolicy(
  val firstProgressTimeoutMs: Long = DEFAULT_FIRST_PROGRESS_TIMEOUT_MS,
  val idleProgressTimeoutMs: Long = DEFAULT_IDLE_PROGRESS_TIMEOUT_MS,
) {
  init {
    require(firstProgressTimeoutMs > 0L) { "firstProgressTimeoutMs must be positive" }
    require(idleProgressTimeoutMs > 0L) { "idleProgressTimeoutMs must be positive" }
  }

  companion object {
    const val DEFAULT_FIRST_PROGRESS_TIMEOUT_MS = 120_000L
    const val DEFAULT_IDLE_PROGRESS_TIMEOUT_MS = 45_000L
    val DEFAULT = InferenceWatchdogPolicy()

    /** Compatibility fallback for old direct callers: this is still progress-aware, not total. */
    fun fromLegacyTimeoutSeconds(timeoutSeconds: Long): InferenceWatchdogPolicy {
      val timeoutMs = timeoutSeconds.coerceAtLeast(1L) * 1000L
      return InferenceWatchdogPolicy(timeoutMs, timeoutMs)
    }
  }
}

/** Small deterministic state machine used by the gateway and focused watchdog tests. */
class InferenceProgressWatchdog(
  private val policy: InferenceWatchdogPolicy,
  private val clock: () -> Long,
) {
  private var firstProgressAtMs = 0L
  private var lastProgressAtMs = clock()
  private var progressEvents = 0
  private var hasProgress = false

  @Synchronized
  fun noteProgress(nowMs: Long = clock()): Boolean {
    if (firstProgressAtMs == 0L) firstProgressAtMs = nowMs
    lastProgressAtMs = nowMs
    progressEvents++
    hasProgress = true
    return true
  }

  @Synchronized
  fun timeoutKind(nowMs: Long = clock()): TimeoutKind {
    if (!hasProgress) {
      return if (nowMs - lastProgressAtMs >= policy.firstProgressTimeoutMs) {
        TimeoutKind.STARTUP
      } else {
        TimeoutKind.NONE
      }
    }
    return if (nowMs - lastProgressAtMs >= policy.idleProgressTimeoutMs) {
      TimeoutKind.IDLE
    } else {
      TimeoutKind.NONE
    }
  }

  @Synchronized fun firstProgressAtMs(): Long = firstProgressAtMs
  @Synchronized fun lastProgressAtMs(): Long = lastProgressAtMs
  @Synchronized fun progressEvents(): Int = progressEvents
}

/** Provenance for a response token count. */
@Serializable
enum class OutputTokenCountSource {
  NATIVE_EXACT,
  SDK_EXACT,
  DERIVED,
  ESTIMATED,
  UNAVAILABLE,
}

/** Conservative evidence about whether the output ceiling was reached. */
@Serializable
enum class OutputLimitEvidence {
  OUTPUT_LIMIT_CONFIRMED,
  OUTPUT_LIMIT_STRONGLY_INDICATED,
  NATIVE_DONE_BELOW_OUTPUT_LIMIT,
  UNKNOWN,
}

/**
 * Bounded, content-free completion evidence for one inference request.
 *
 * This intentionally contains no prompt or generated text.  Request identity and model name
 * are supplied by the owning request log entry, while this object records only terminal and
 * timing metadata needed to investigate premature completion.
 */
@Serializable
data class InferenceCompletionDiagnostic(
  val nativeTerminalSignal: NativeTerminalSignal = NativeTerminalSignal.UNAVAILABLE,
  val gatewayTerminalOutcome: GatewayTerminalOutcome = GatewayTerminalOutcome.UNAVAILABLE,
  val timeoutKind: TimeoutKind = TimeoutKind.NONE,
  val sdkFinishReason: String? = null,
  val sdkFinishReasonAvailable: Boolean = false,
  val effectiveMaxOutputTokens: Int = 0,
  val engineContextTokens: Int = 0,
  val modelMaxContextTokens: Int = 0,
  val usableInputTokens: Int = 0,
  val safetyReserveTokens: Int = 0,
  val timeoutConfiguredMs: Long = 0L,
  val firstProgressTimeoutMs: Long = 0L,
  val idleProgressTimeoutMs: Long = 0L,
  val backend: String? = null,
  val nativeContextBefore: Long = 0L,
  val nativeContextAfter: Long = 0L,
  val outputTokenCount: Long = 0L,
  val outputTokenCountSource: OutputTokenCountSource = OutputTokenCountSource.UNAVAILABLE,
  val outputLimitEvidence: OutputLimitEvidence = OutputLimitEvidence.UNKNOWN,
  val outputCharCount: Int = 0,
  val streamChunkCount: Int = 0,
  val startedAtElapsedMs: Long = 0L,
  val firstTokenAtElapsedMs: Long = 0L,
  val lastChunkAtElapsedMs: Long = 0L,
  val nativeDoneAtElapsedMs: Long = 0L,
  val settledAtElapsedMs: Long = 0L,
  val totalLatencyMs: Long = 0L,
  val lastChunkToDoneMs: Long = 0L,
  val firstProgressAtElapsedMs: Long = 0L,
  val lastProgressAtElapsedMs: Long = 0L,
  val progressEventCount: Int = 0,
  val lastProgressAgeMsAtTermination: Long = 0L,
  val contextCompacted: Boolean = false,
  val overflowRetryAttempted: Boolean = false,
)

/**
 * Thread-safe evidence collector shared by native callbacks and gateway settlement.
 * Native and gateway terminal signals are independently first-writer-wins, so a late callback
 * cannot overwrite the signal that actually arrived first or create a second terminal record.
 */
class InferenceCompletionEvidence(
  private val startedAtElapsedMs: Long,
  private val clock: () -> Long = { SystemClock.elapsedRealtime() },
) {
  private val nativeSignal = AtomicReference(NativeTerminalSignal.UNAVAILABLE)
  private val gatewayOutcome = AtomicReference(GatewayTerminalOutcome.UNAVAILABLE)
  private val chunkCount = AtomicInteger(0)
  private val firstTokenAt = AtomicLong(0L)
  private val lastChunkAt = AtomicLong(0L)
  private val firstProgressAt = AtomicLong(0L)
  private val lastProgressAt = AtomicLong(0L)
  private val progressEventCount = AtomicInteger(0)
  private val timeoutKind = AtomicReference(TimeoutKind.NONE)
  private val nativeDoneAt = AtomicLong(0L)
  private val terminalRecorded = AtomicBoolean(false)

  fun recordNativeSignal(signal: NativeTerminalSignal): Boolean {
    val accepted = nativeSignal.compareAndSet(NativeTerminalSignal.UNAVAILABLE, signal)
    if (accepted && signal == NativeTerminalSignal.ON_DONE) {
      nativeDoneAt.compareAndSet(0L, clock())
    }
    return accepted
  }

  fun recordGatewayOutcome(outcome: GatewayTerminalOutcome): Boolean =
    gatewayOutcome.compareAndSet(GatewayTerminalOutcome.UNAVAILABLE, outcome)

  /** Records native output before it is forwarded to any HTTP/UI consumer. */
  fun recordNativeProgress(nowMs: Long = clock()): Boolean {
    firstProgressAt.compareAndSet(0L, nowMs)
    firstTokenAt.compareAndSet(0L, nowMs)
    lastProgressAt.set(nowMs)
    progressEventCount.incrementAndGet()
    return true
  }

  fun recordTimeoutKind(kind: TimeoutKind): Boolean =
    timeoutKind.compareAndSet(TimeoutKind.NONE, kind)

  fun recordChunk(nonEmpty: Boolean) {
    if (!nonEmpty) return
    lastChunkAt.set(clock())
    chunkCount.incrementAndGet()
  }

  /** Builds exactly one terminal snapshot for this request. */
  fun snapshot(
    outputCharCount: Int,
    outputTokenCount: Long,
    outputTokenCountSource: OutputTokenCountSource,
    effectiveMaxOutputTokens: Int,
    engineContextTokens: Int = 0,
    modelMaxContextTokens: Int = 0,
    usableInputTokens: Int = 0,
    safetyReserveTokens: Int = 0,
    timeoutConfiguredMs: Long = 0L,
    firstProgressTimeoutMs: Long = 0L,
    idleProgressTimeoutMs: Long = 0L,
    backend: String? = null,
    nativeContextBefore: Long,
    nativeContextAfter: Long,
    contextCompacted: Boolean,
    overflowRetryAttempted: Boolean = false,
  ): InferenceCompletionDiagnostic? {
    if (!terminalRecorded.compareAndSet(false, true)) return null
    val settledAt = clock()
    val doneAt = nativeDoneAt.get()
    val firstProgress = firstProgressAt.get()
    val lastProgress = lastProgressAt.get()
    val lastChunk = lastProgress.takeIf { it > 0L } ?: lastChunkAt.get()
    val effectiveIdleTimeout = idleProgressTimeoutMs.takeIf { it > 0L } ?: timeoutConfiguredMs
    return InferenceCompletionDiagnostic(
      nativeTerminalSignal = nativeSignal.get(),
      gatewayTerminalOutcome = gatewayOutcome.get(),
      timeoutKind = timeoutKind.get(),
      effectiveMaxOutputTokens = effectiveMaxOutputTokens.coerceAtLeast(0),
      engineContextTokens = engineContextTokens.coerceAtLeast(0),
      modelMaxContextTokens = modelMaxContextTokens.coerceAtLeast(0),
      usableInputTokens = usableInputTokens.coerceAtLeast(0),
      safetyReserveTokens = safetyReserveTokens.coerceAtLeast(0),
      timeoutConfiguredMs = effectiveIdleTimeout.coerceAtLeast(0L),
      firstProgressTimeoutMs = firstProgressTimeoutMs.coerceAtLeast(0L),
      idleProgressTimeoutMs = effectiveIdleTimeout.coerceAtLeast(0L),
      backend = backend?.take(32),
      nativeContextBefore = nativeContextBefore.coerceAtLeast(0L),
      nativeContextAfter = nativeContextAfter.coerceAtLeast(0L),
      outputTokenCount = outputTokenCount.coerceAtLeast(0L),
      outputTokenCountSource = outputTokenCountSource,
      outputLimitEvidence = classifyOutputLimitEvidence(
        nativeSignal = nativeSignal.get(),
        sdkFinishReason = null,
        sdkFinishReasonAvailable = false,
        outputTokenCount = outputTokenCount,
        outputTokenCountSource = outputTokenCountSource,
        effectiveMaxOutputTokens = effectiveMaxOutputTokens,
      ),
      outputCharCount = outputCharCount.coerceAtLeast(0),
      streamChunkCount = chunkCount.get(),
      startedAtElapsedMs = startedAtElapsedMs,
      firstTokenAtElapsedMs = firstTokenAt.get(),
      lastChunkAtElapsedMs = lastChunk,
      nativeDoneAtElapsedMs = doneAt,
      settledAtElapsedMs = settledAt,
      totalLatencyMs = (settledAt - startedAtElapsedMs).coerceAtLeast(0L),
      lastChunkToDoneMs = if (lastChunk > 0L && doneAt >= lastChunk) doneAt - lastChunk else 0L,
      firstProgressAtElapsedMs = firstProgress,
      lastProgressAtElapsedMs = lastProgress,
      progressEventCount = progressEventCount.get(),
      lastProgressAgeMsAtTermination = if (lastProgress > 0L) {
        (settledAt - lastProgress).coerceAtLeast(0L)
      } else {
        0L
      },
      contextCompacted = contextCompacted,
      overflowRetryAttempted = overflowRetryAttempted,
    )
  }

  fun nativeTerminalSignal(): NativeTerminalSignal = nativeSignal.get()
}

/**
 * Do not infer EOS from an ordinary onDone callback.  This classifier only reports output-limit
 * evidence when the SDK provides a reason or an exact/SDK token count reaches the configured cap.
 */
fun classifyOutputLimitEvidence(
  nativeSignal: NativeTerminalSignal,
  sdkFinishReason: String?,
  sdkFinishReasonAvailable: Boolean,
  outputTokenCount: Long,
  outputTokenCountSource: OutputTokenCountSource,
  effectiveMaxOutputTokens: Int,
): OutputLimitEvidence {
  if (sdkFinishReasonAvailable && sdkFinishReason != null) {
    val reason = sdkFinishReason.lowercase()
    if (reason.contains("length") || reason.contains("max") || reason.contains("limit")) {
      return OutputLimitEvidence.OUTPUT_LIMIT_CONFIRMED
    }
  }
  val exactCount = outputTokenCountSource == OutputTokenCountSource.NATIVE_EXACT ||
    outputTokenCountSource == OutputTokenCountSource.SDK_EXACT
  if (exactCount && effectiveMaxOutputTokens > 0 && outputTokenCount >= effectiveMaxOutputTokens) {
    return OutputLimitEvidence.OUTPUT_LIMIT_STRONGLY_INDICATED
  }
  if (nativeSignal == NativeTerminalSignal.ON_DONE &&
    effectiveMaxOutputTokens > 0 &&
    outputTokenCountSource != OutputTokenCountSource.UNAVAILABLE &&
    outputTokenCount < effectiveMaxOutputTokens
  ) {
    return OutputLimitEvidence.NATIVE_DONE_BELOW_OUTPUT_LIMIT
  }
  return OutputLimitEvidence.UNKNOWN
}

/** Single-line, content-free marker used to correlate a settled request in Logcat. */
fun completionDiagnosticMarker(
  requestId: String,
  modelName: String?,
  diagnostic: InferenceCompletionDiagnostic,
): String = buildString {
  append("FLOWASSIST_COMPLETION ")
  append("requestId=").append(requestId.take(80)).append(' ')
  append("model=").append(modelName?.take(80) ?: "UNKNOWN").append(' ')
  append("nativeSignal=").append(diagnostic.nativeTerminalSignal).append(' ')
  append("gatewayOutcome=").append(diagnostic.gatewayTerminalOutcome).append(' ')
  append("timeoutKind=").append(diagnostic.timeoutKind).append(' ')
  append("sdkFinishReason=").append(diagnostic.sdkFinishReason ?: "UNAVAILABLE").append(' ')
  append("engineContext=").append(diagnostic.engineContextTokens).append(' ')
  append("modelMaxContext=").append(diagnostic.modelMaxContextTokens).append(' ')
  append("usableInput=").append(diagnostic.usableInputTokens).append(' ')
  append("maxOutput=").append(diagnostic.effectiveMaxOutputTokens).append(' ')
  append("timeoutMs=").append(diagnostic.timeoutConfiguredMs).append(' ')
  append("firstProgressTimeoutMs=").append(diagnostic.firstProgressTimeoutMs).append(' ')
  append("idleProgressTimeoutMs=").append(diagnostic.idleProgressTimeoutMs).append(' ')
  append("backend=").append(diagnostic.backend ?: "UNAVAILABLE").append(' ')
  append("outputTokens=").append(diagnostic.outputTokenCount).append(' ')
  append("outputTokenSource=").append(diagnostic.outputTokenCountSource).append(' ')
  append("outputLimitEvidence=").append(diagnostic.outputLimitEvidence).append(' ')
  append("nativeBefore=").append(diagnostic.nativeContextBefore).append(' ')
  append("nativeAfter=").append(diagnostic.nativeContextAfter).append(' ')
  append("chunks=").append(diagnostic.streamChunkCount).append(' ')
  append("progressEvents=").append(diagnostic.progressEventCount).append(' ')
  append("firstProgressAt=").append(diagnostic.firstProgressAtElapsedMs).append(' ')
  append("lastProgressAt=").append(diagnostic.lastProgressAtElapsedMs).append(' ')
  append("lastProgressAgeMs=").append(diagnostic.lastProgressAgeMsAtTermination).append(' ')
  append("latencyMs=").append(diagnostic.totalLatencyMs).append(' ')
  append("lastChunkToDoneMs=").append(diagnostic.lastChunkToDoneMs)
}
