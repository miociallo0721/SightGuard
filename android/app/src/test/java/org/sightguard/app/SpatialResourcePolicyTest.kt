package org.sightguard.app

import org.junit.Assert.*
import org.junit.Test

class SpatialResourcePolicyTest {
    @Test fun degradedSemanticsCanConfirmBeforeRecoveryExpiresWhileCriticalLoadPauses() {
        val config=EnvironmentConfig()
        val normal=SpatialResourcePolicy.plan(0,false,0,config.sampleIntervalMs)
        assertTrue(normal.depthAllowed);assertFalse(normal.paused)
        for(level in listOf(2,3)) {
            val reduced=SpatialResourcePolicy.plan(level,false,0,config.sampleIntervalMs)
            assertFalse(reduced.paused);assertFalse(reduced.depthAllowed)
            assertTrue(reduced.intervalMs>normal.intervalMs)
            assertTrue("Two fresh observations must fit the event recovery window",reduced.intervalMs<config.recoveryMs)
        }
        assertTrue(SpatialResourcePolicy.plan(4,false,0,config.sampleIntervalMs).paused)
        assertTrue(SpatialResourcePolicy.plan(0,true,0,config.sampleIntervalMs).paused)
        assertTrue(SpatialResourcePolicy.plan(0,false,10,config.sampleIntervalMs).paused)
        val slow=SpatialResourcePolicy.plan(0,false,5,config.sampleIntervalMs)
        assertFalse(slow.depthAllowed);assertTrue(slow.intervalMs<config.recoveryMs)
    }
}
