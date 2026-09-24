package com.ollitert.llm.server.service.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class InferenceCompletionDiagnosticsTest {
  @Test
  fun nativeOnDoneRemainsDistinctAndSettlesOnce() {
    var now = 100L
    val evidence = InferenceCompletionEvidence(100L) { now }

    assertTrue(evidence.recordNativeSignal(NativeTerminalSignal.ON_DONE))
    assertFalse(evidence.recordNativeSignal(NativeTerminalSignal.ON_ERROR))
    evidence.recordGatewayOutcome(GatewayTerminalOutcome.SUCCESS)
    evidence.recordChunk(nonEmpty = true)
    now = 250L

    val diagnostic = evidence.snapshot(
      outputCharCount = 120,
      outputTokenCount = 30,
      outputTokenCountSource = OutputTokenCountSource.ESTIMATED,
      effectiveMaxOutputTokens = 4_000,
      nativeContextBefore = 4_000,
      nativeContextAfter = 4_100,
    )

    assertNotNull(diagnostic)
    assertEquals(NativeTerminalSignal.ON_DONE, diagnostic?.nativeTerminalSignal)
    assertEquals(GatewayTerminalOutcome.SUCCESS, diagnostic?.gatewayTerminalOutcome)
    assertEquals(OutputLimitEvidence.NATIVE_DONE_BELOW_OUTPUT_LIMIT, diagnostic?.outputLimitEvidence)
    assertNull(
      evidence.snapshot(
        outputCharCount = 120,
        outputTokenCount = 30,
        outputTokenCountSource = OutputTokenCountSource.ESTIMATED,
        effectiveMaxOutputTokens = 4_000,
        nativeContextBefore = 4_000,
        nativeContextAfter = 4_100,
      ),
    )
  }

  @Test
  fun cancellationExceptionIsNotRecordedAsOnDone() {
    val evidence = InferenceCompletionEvidence(10L) { 20L }
    evidence.recordNativeSignal(NativeTerminalSignal.CANCELLATION_EXCEPTION)
    evidence.recordGatewayOutcome(GatewayTerminalOutcome.SUCCESS)

    val diagnostic = evidence.snapshot(
      outputCharCount = 0,
      outputTokenCount = 0,
      outputTokenCountSource = OutputTokenCountSource.UNAVAILABLE,
      effectiveMaxOutputTokens = 4_000,
    )

    assertEquals(NativeTerminalSignal.CANCELLATION_EXCEPTION, diagnostic?.nativeTerminalSignal)
    assertEquals(GatewayTerminalOutcome.SUCCESS, diagnostic?.gatewayTerminalOutcome)
  }

  @Test
  fun nativeErrorAndGatewayErrorRemainObservable() {
    val evidence = InferenceCompletionEvidence(0L) { 50L }
    evidence.recordNativeSignal(NativeTerminalSignal.ON_ERROR)
    evidence.recordGatewayOutcome(GatewayTerminalOutcome.ERROR)

    val diagnostic = evidence.snapshot(
      outputCharCount = 10,
      outputTokenCount = 3,
      outputTokenCountSource = OutputTokenCountSource.ESTIMATED,
      effectiveMaxOutputTokens = 4_000,
    )

    assertEquals(NativeTerminalSignal.ON_ERROR, diagnostic?.nativeTerminalSignal)
    assertEquals(GatewayTerminalOutcome.ERROR, diagnostic?.gatewayTerminalOutcome)
  }

  @Test
  fun timeoutIsNotClassifiedAsNativeDone() {
    val evidence = InferenceCompletionEvidence(0L) { 1_000L }
    evidence.recordGatewayOutcome(GatewayTerminalOutcome.TIMEOUT)

    val diagnostic = evidence.snapshot(
      outputCharCount = 100,
      outputTokenCount = 25,
      outputTokenCountSource = OutputTokenCountSource.ESTIMATED,
      effectiveMaxOutputTokens = 4_000,
    )

    assertEquals(NativeTerminalSignal.UNAVAILABLE, diagnostic?.nativeTerminalSignal)
    assertEquals(GatewayTerminalOutcome.TIMEOUT, diagnostic?.gatewayTerminalOutcome)
    assertEquals(OutputLimitEvidence.UNKNOWN, diagnostic?.outputLimitEvidence)
  }

  @Test
  fun exactCapIsEvidenceButEstimatedCountIsNot() {
    assertEquals(
      OutputLimitEvidence.OUTPUT_LIMIT_STRONGLY_INDICATED,
      classifyOutputLimitEvidence(
        nativeSignal = NativeTerminalSignal.ON_DONE,
        sdkFinishReason = null,
        sdkFinishReasonAvailable = false,
        outputTokenCount = 4_000,
        outputTokenCountSource = OutputTokenCountSource.SDK_EXACT,
        effectiveMaxOutputTokens = 4_000,
      ),
    )
    assertEquals(
      OutputLimitEvidence.NATIVE_DONE_BELOW_OUTPUT_LIMIT,
      classifyOutputLimitEvidence(
        nativeSignal = NativeTerminalSignal.ON_DONE,
        sdkFinishReason = null,
        sdkFinishReasonAvailable = false,
        outputTokenCount = 400,
        outputTokenCountSource = OutputTokenCountSource.ESTIMATED,
        effectiveMaxOutputTokens = 4_000,
      ),
    )
  }
}
