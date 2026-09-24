package com.ollitert.llm.server.service.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class InferenceProgressWatchdogTest {
  private val policy = InferenceWatchdogPolicy(
    firstProgressTimeoutMs = 120_000L,
    idleProgressTimeoutMs = 45_000L,
  )

  @Test
  fun healthyGenerationMayContinuePastLegacyTotalDeadline() {
    val watchdog = InferenceProgressWatchdog(policy) { 0L }

    watchdog.noteProgress(10_000L)
    assertEquals(TimeoutKind.NONE, watchdog.timeoutKind(25_000L))
    watchdog.noteProgress(25_000L)
    assertEquals(TimeoutKind.NONE, watchdog.timeoutKind(45_000L))
    watchdog.noteProgress(45_000L)
    assertEquals(TimeoutKind.NONE, watchdog.timeoutKind(70_000L))
    watchdog.noteProgress(70_000L)
    assertEquals(TimeoutKind.NONE, watchdog.timeoutKind(105_000L))
    watchdog.noteProgress(105_000L)
    assertEquals(TimeoutKind.NONE, watchdog.timeoutKind(145_000L))
    watchdog.noteProgress(145_000L)
    assertEquals(TimeoutKind.NONE, watchdog.timeoutKind(180_000L))
    assertEquals(5, watchdog.progressEvents())
  }

  @Test
  fun noInitialProgressProducesStartupTimeout() {
    val watchdog = InferenceProgressWatchdog(policy) { 1_000L }

    assertEquals(TimeoutKind.NONE, watchdog.timeoutKind(120_999L))
    assertEquals(TimeoutKind.STARTUP, watchdog.timeoutKind(121_000L))
  }

  @Test
  fun stalledStreamProducesIdleTimeout() {
    val watchdog = InferenceProgressWatchdog(policy) { 1_000L }

    watchdog.noteProgress(10_000L)
    watchdog.noteProgress(20_000L)
    assertEquals(TimeoutKind.NONE, watchdog.timeoutKind(64_999L))
    assertEquals(TimeoutKind.IDLE, watchdog.timeoutKind(65_000L))
  }

  @Test
  fun reasoningOnlyProgressRefreshesWatchdog() {
    val watchdog = InferenceProgressWatchdog(policy) { 1_000L }

    watchdog.noteProgress(10_000L)
    watchdog.noteProgress(50_000L)

    assertEquals(TimeoutKind.NONE, watchdog.timeoutKind(94_999L))
    assertTrue(watchdog.progressEvents() == 2)
  }
}
