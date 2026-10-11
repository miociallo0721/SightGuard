package org.sightguard.app

data class SpatialResourcePlan(val paused: Boolean, val intervalMs: Long, val depthAllowed: Boolean, val reason: String)

/** Android thermal levels 0..6: reduce optional work before suspending semantics. */
object SpatialResourcePolicy {
    fun plan(thermalLevel: Int, lowMemory: Boolean, slowPrimaryFrames: Int, baseIntervalMs: Long): SpatialResourcePlan = when {
        lowMemory -> SpatialResourcePlan(true,6000,false,"low_memory")
        thermalLevel>=4 -> SpatialResourcePlan(true,6000,false,"thermal_critical")
        slowPrimaryFrames>=10 -> SpatialResourcePlan(true,6000,false,"primary_overloaded")
        slowPrimaryFrames>=5 -> SpatialResourcePlan(false,maxOf(baseIntervalMs,3000),false,"primary_slow")
        thermalLevel>=3 -> SpatialResourcePlan(false,maxOf(baseIntervalMs,2000),false,"thermal_severe_semantics_only")
        thermalLevel>=2 -> SpatialResourcePlan(false,maxOf(baseIntervalMs,1300),false,"thermal_moderate_semantics_only")
        else -> SpatialResourcePlan(false,baseIntervalMs,true,"normal")
    }
}
