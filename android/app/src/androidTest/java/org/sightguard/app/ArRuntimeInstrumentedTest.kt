package org.sightguard.app

import androidx.test.runner.lifecycle.ActivityLifecycleMonitorRegistry
import androidx.test.runner.lifecycle.Stage
import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File

/** Device pipeline/lifecycle test. Tracking/depth observations are recorded, not presumed valid. */
class ArRuntimeInstrumentedTest {
    @Test fun installedArRuntimeProducesPrimaryFramesAndReleases() = runCamera(true)
    @Test fun disabledArFallsBackToCameraXAndReleases() = runCamera(false)
    private fun runCamera(ar: Boolean) {
        val ins=InstrumentationRegistry.getInstrumentation();val context=ins.targetContext
        val prefs=context.getSharedPreferences("perception",0)
        val oldAuto=prefs.getBoolean("auto_start",true);val oldAr=prefs.getBoolean("ar_enabled",true);val oldMute=prefs.getBoolean("muted",false)
        prefs.edit().putBoolean("auto_start",true).putBoolean("ar_enabled",ar).putBoolean("muted",true).commit()
        var activity: MainActivity?=null;val observations=JSONArray()
        fun state(): JSONObject {var s=JSONObject();ins.runOnMainSync {s=activity!!.sessionDiagnostics()};return s}
        try {
            // Launch through the authorized ADB instrumentation shell identity. HyperOS
            // rejects application-origin background activity starts from the test runner.
            android.os.ParcelFileDescriptor.AutoCloseInputStream(ins.uiAutomation.executeShellCommand("am start -n ${context.packageName}/.MainActivity")).use {it.readBytes()}
            val launchDeadline=SystemClock.elapsedRealtime()+10000
            while(activity==null && SystemClock.elapsedRealtime()<launchDeadline) {
                ins.runOnMainSync {activity=ActivityLifecycleMonitorRegistry.getInstance().getActivitiesInStage(Stage.RESUMED).filterIsInstance<MainActivity>().firstOrNull()}
                SystemClock.sleep(100)
            }
            assertNotNull("ADB activity launch failed",activity)
            val end=SystemClock.elapsedRealtime()+15000
            while(SystemClock.elapsedRealtime()<end) {observations.put(state());SystemClock.sleep(500)}
            val latest=state()
            assertTrue("Primary camera inference stalled: $latest",latest.getInt("frameIndex")>=15)
            assertEquals(ar,latest.getBoolean("arActive"))
            assertTrue(latest.getBoolean("modelReady"))
            ins.runOnMainSync {activity!!.finish()};ins.waitForIdleSync();SystemClock.sleep(1500)
            val stopped=state();observations.put(stopped)
            assertFalse(stopped.getBoolean("running"));assertFalse(stopped.getBoolean("arActive"));assertFalse(stopped.getBoolean("modelReady"))
        } finally {
            activity?.let {a ->ins.runOnMainSync {if(!a.isFinishing) a.finish()}};ins.waitForIdleSync()
            prefs.edit().putBoolean("auto_start",oldAuto).putBoolean("ar_enabled",oldAr).putBoolean("muted",oldMute).commit()
            File(context.filesDir,"ar-runtime-$ar.json").writeText(JSONObject().put("scope","actual_camera_pipeline_not_measured_wall_distance_or_motion_acceptance")
                .put("device",android.os.Build.MODEL).put("ar_enabled",ar).put("samples",observations).toString(2))
        }
    }
}
