package org.sightguard.app

import android.Manifest
import android.content.Intent
import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.json.JSONObject

/** Real CameraX cold-start checks, separate from the pure intent-state fixtures. */
class AutoStartInstrumentedTest {
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
    private fun state(activity: MainActivity): JSONObject { var state=JSONObject(); instrumentation.runOnMainSync { state=activity.sessionDiagnostics() }; return state }
    private fun launch(enabled:Boolean): MainActivity {
        val context=instrumentation.targetContext
        context.getSharedPreferences("perception",0).edit().putBoolean("auto_start",enabled).commit()
        android.os.ParcelFileDescriptor.AutoCloseInputStream(instrumentation.uiAutomation.executeShellCommand("pm grant ${context.packageName} ${Manifest.permission.CAMERA}")).use { it.readBytes() }
        return instrumentation.startActivitySync(Intent(context,MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) as MainActivity
    }
    @Test fun coldStartRunsActualCameraInferenceAndStops() {
        val activity=launch(true)
        try {
            val deadline=SystemClock.elapsedRealtime()+15000
            while(state(activity).optInt("frameIndex",-1)<1 && SystemClock.elapsedRealtime()<deadline) SystemClock.sleep(100)
            val result=state(activity)
            assertTrue("Actual CameraX frame required: $result",result.getInt("frameIndex")>=1)
            assertTrue(result.getBoolean("modelReady")); assertEquals("camera",result.getString("source"))
        } finally { instrumentation.runOnMainSync{activity.finish()}; instrumentation.waitForIdleSync() }
    }
    @Test fun disabledPreferenceDoesNotStartOnColdLaunch() {
        val activity=launch(false)
        try { SystemClock.sleep(1000); val result=state(activity); assertFalse(result.getBoolean("running")); assertFalse(result.getBoolean("modelReady")); assertEquals(-1,result.getInt("frameIndex")) }
        finally { instrumentation.runOnMainSync{activity.finish()}; instrumentation.targetContext.getSharedPreferences("perception",0).edit().putBoolean("auto_start",true).commit(); instrumentation.waitForIdleSync() }
    }
}
