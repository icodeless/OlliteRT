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
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ContextCapacityPlannerTest {

    private val healthyMemory = ContextMemoryView(
        totalBytes = 8L * 1024 * 1024 * 1024,
        availableBytes = 6L * 1024 * 1024 * 1024,
        lowMemory = false,
    )

    private fun plan(
        modelMaxContextTokens: Int = 32_768,
        maxOutputTokens: Int = 2_048,
        backend: Backend = Backend.CPU(),
        memory: ContextMemoryView? = null,
        preferredTier: Int? = null,
        modelSizeBytes: Long = 1_000_000_000L,
        modelMinDeviceMemoryInGb: Int? = null,
    ) = ContextCapacityPlanner.plan(
        modelMaxContextTokens = modelMaxContextTokens,
        maxOutputTokens = maxOutputTokens,
        backend = backend,
        memory = memory,
        preferredTier = preferredTier,
        modelSizeBytes = modelSizeBytes,
        modelMinDeviceMemoryInGb = modelMinDeviceMemoryInGb,
    )

    @Test
    fun cachedLowTierAllowsUpwardRecoveryWhenMemoryPermits() {
        assertEquals(
            listOf(32_000, 24_000, 16_000, 8_000),
            ContextCapacityPlanner.orderCandidateTiers(
                tiers = listOf(8_000, 16_000, 24_000, 32_000),
                cached = 16_000,
                allowUpwardRecovery = true,
            ),
        )
    }

    @Test
    fun cachedTierRemainsFastPathWhenMemoryDoesNotPermitUpgrade() {
        assertEquals(
            listOf(16_000, 8_000),
            ContextCapacityPlanner.orderCandidateTiers(
                tiers = listOf(16_000, 8_000),
                cached = 16_000,
                allowUpwardRecovery = false,
            ),
        )
    }

    @Test
    fun plannerIsPureAndDoesNotDependOnAmbientState() {
        // Two identical calls must produce identical plans. This is the property that makes the
        // planner testable in the substrate: nothing is read from disk or ambient state.
        assertEquals(plan(), plan())
    }

    @Test
    fun nullMemoryFallsBackToNativeProbing() {
        val result = plan(memory = null)
        assertEquals("native_probe", result.source)
        assertEquals(32_768, result.engineContextTokens)
    }

    @Test
    fun declaredContextIsNeverExceededByTheOutputReserve() {
        val result = plan(modelMaxContextTokens = 2_048, maxOutputTokens = 8_192)
        assertTrue(
            "output reserve must not push the engine past the declared capacity",
            result.maxOutputTokens <= 2_048 - 1_000 - ContextCapacityPlanner.MIN_INPUT_TOKENS,
        )
        assertTrue(result.maxOutputTokens >= 1)
    }

    @Test
    fun usableInputIsSelectedMinusOutputAndReserve() {
        val result = plan(maxOutputTokens = 1_024, memory = healthyMemory)
        assertEquals(
            (result.engineContextTokens - result.maxOutputTokens - result.safetyReserveTokens)
                .coerceAtLeast(1),
            result.usableInputTokens,
        )
    }

    @Test
    fun lowMemoryCapsTheContextAllocation() {
        val lowMemory = ContextMemoryView(
            totalBytes = 4L * 1024 * 1024 * 1024,
            availableBytes = 512L * 1024 * 1024,
            lowMemory = true,
        )
        val result = plan(memory = lowMemory)
        assertTrue(
            "low memory must cap the allocation, got ${result.engineContextTokens}",
            result.engineContextTokens <= 8_192,
        )
        assertEquals("memory_adaptive", result.source)
    }

    @Test
    fun preferredTierOutsideLegalCandidatesIsIgnored() {
        // A remembered tier that current conditions no longer permit must not be trusted.
        val lowMemory = ContextMemoryView(
            totalBytes = 4L * 1024 * 1024 * 1024,
            availableBytes = 512L * 1024 * 1024,
            lowMemory = true,
        )
        val result = plan(memory = lowMemory, preferredTier = 32_768)
        assertTrue(result.engineContextTokens <= 8_192)
        assertEquals("memory_adaptive", result.source)
    }

    @Test
    fun rememberedTierLeadsTheOrderWhenMemoryCapsTheAllocation() {
        // Low memory caps the allocation at 8192, so no tier above the remembered one is legal.
        // With no upward recovery possible, the remembered tier is tried first.
        val capped = ContextMemoryView(
            totalBytes = 4L * 1024 * 1024 * 1024,
            availableBytes = 512L * 1024 * 1024,
            lowMemory = true,
        )
        val result = plan(memory = capped, preferredTier = 8_192)
        assertEquals("cached", result.source)
        assertEquals(8_192, result.candidateTiers.first())
    }

    @Test
    fun rememberedTierDoesNotCeilTheAllocationWhenMemoryPermitsRecovery() {
        // The remembered tier is a hint, not a ceiling. With ample memory the planner must try
        // larger tiers first so a temporary low-memory downgrade can recover naturally.
        val result = plan(memory = healthyMemory, preferredTier = 8_192)
        assertEquals("cached", result.source)
        assertTrue(
            "upward recovery must reorder above the remembered tier, got ${result.candidateTiers}",
            result.candidateTiers.first() > 8_192,
        )
        assertTrue(8_192 in result.candidateTiers)
    }

    @Test
    fun candidateTiersAreDescendingAndWithinDeclaredCapacity() {
        val result = plan(memory = healthyMemory)
        assertEquals(result.candidateTiers.sortedDescending(), result.candidateTiers)
        assertTrue(result.candidateTiers.all { it <= result.modelMaxContextTokens })
    }

    @Test
    fun modelRequiringMoreRamThanDeviceHasIsCapped() {
        val modest = ContextMemoryView(
            totalBytes = 2L * 1024 * 1024 * 1024,
            availableBytes = 1536L * 1024 * 1024,
            lowMemory = false,
        )
        val result = plan(memory = modest, modelMinDeviceMemoryInGb = 8)
        assertTrue(
            "device below the model's minimum must be capped, got ${result.engineContextTokens}",
            result.engineContextTokens <= 16_384,
        )
    }

    @Test
    fun noOpStoreIsPermissiveAndRemembersNothing() {
        val store = NoOpContextCapacityTierStore
        assertNull(store.readPreferredTier("m", "v1", Backend.CPU()))
        // Recording through the no-op must not throw; it simply discards.
        store.recordSuccessfulTier("m", "v1", Backend.CPU(), 8_192)
    }
}
