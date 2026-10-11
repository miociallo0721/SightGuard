package org.sightguard.app

/** Foreground-only intent state. Compose never drives session starts. */
class AutoStartPolicy(var enabled: Boolean = true, var explicitlyStopped: Boolean = false) {
    var foreground = false
        private set
    private var cold = true
    private var resumeWanted = false
    private var attempted = false
    var fileMode = false
    fun enter(permission: Boolean): String? {
        foreground = true
        val wanted = enabled && !explicitlyStopped && !fileMode && (cold || resumeWanted)
        cold = false; resumeWanted = false
        if (!wanted || attempted) return null
        attempted = true
        return if (permission) "START" else "PERMISSION"
    }
    fun leave(running: Boolean, requestedRunning: Boolean = false) { resumeWanted = (running || requestedRunning) && enabled && !explicitlyStopped && !fileMode; foreground = false; if (resumeWanted) attempted = false }
    fun manualStart() { explicitlyStopped = false; attempted = true }
    fun manualStop() { explicitlyStopped = true; resumeWanted = false }
    fun sourceChanged(isFile: Boolean) { fileMode = isFile; resumeWanted = false; attempted = true }
    fun permissionGranted(manual: Boolean): Boolean = foreground && !explicitlyStopped && !fileMode && (manual || enabled)
    fun failed() { resumeWanted = false; attempted = true }
}
