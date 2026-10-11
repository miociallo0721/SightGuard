package org.sightguard.app

import org.junit.Assert.*
import org.junit.Test

class LightTrackerTest {
    private fun person(x: Float) = Detection(0, .9f, x, .3f, x + .1f, .8f)
    @Test fun nonuniformMotionOcclusionAndExpiryKeepOrEndIdentity() {
        val tracker = LightTracker()
        val first = tracker.update(listOf(person(.1f)), 0).single().trackId
        assertEquals(first, tracker.update(listOf(person(.11f)), 100).single().trackId)
        assertEquals(first, tracker.update(listOf(person(.14f)), 400).single().trackId)
        assertTrue(tracker.update(emptyList(), 500).isEmpty())
        assertEquals(first, tracker.update(listOf(person(.15f)), 1500).single().trackId)
        assertNotEquals(first, tracker.update(listOf(person(.15f)), 5000).single().trackId)
        tracker.reset()
        assertEquals(1L, tracker.update(listOf(person(.15f)), 0).single().trackId)
    }
    @Test fun duplicateRowsDoNotCreateTwoRiskTracksButNearbyDistinctPeopleRemain() {
        val tracker = LightTracker()
        assertEquals(2, tracker.update(listOf(person(.1f), person(.101f), person(.22f)), 0).size)
    }
    @Test fun ambiguityDoesNotForceTwoPeopleIntoOneIdentity() {
        val tracker = LightTracker()
        val old = tracker.update(listOf(person(.1f), person(.22f)), 0).map { it.trackId }.toSet()
        val ambiguous = tracker.update(listOf(person(.16f)), 100).single()
        assertFalse(ambiguous.trackId in old)
    }
    @Test(expected = IllegalArgumentException::class) fun backwardsTimeRequiresReset() {
        val tracker = LightTracker(); tracker.update(listOf(person(.1f)), 500); tracker.update(listOf(person(.1f)), 100)
    }
}
