package org.sightguard.app

import org.junit.Assert.*
import org.junit.Test

class ResultFreshnessTest {
    @Test fun rejectsExpiredAndFutureResults() {
        assertTrue(ResultFreshness.isCurrent(1000, 2500))
        assertFalse(ResultFreshness.isCurrent(1000, 2501))
        assertFalse(ResultFreshness.isCurrent(1000, 999))
        assertFalse(ResultFreshness.isCurrent(-1, 1000))
    }
}
