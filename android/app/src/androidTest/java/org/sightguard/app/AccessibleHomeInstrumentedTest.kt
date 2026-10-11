package org.sightguard.app

import android.Manifest
import android.content.Intent
import android.graphics.Rect
import android.os.SystemClock
import android.view.accessibility.AccessibilityNodeInfo
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

/** Exercises the platform accessibility tree and actual controls, rather than source string matching. */
class AccessibleHomeInstrumentedTest {
    @Test fun repeatedActualCameraStartStopReleasesModelsAndRecordsMemory() {
        val context=instrumentation.targetContext
        context.getSharedPreferences("perception",0).edit().putBoolean("auto_start",true).putBoolean("muted",false).commit()
        val activity=instrumentation.startActivitySync(Intent(context,MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) as MainActivity
        val cycles=org.json.JSONArray()
        try {
            repeat(3) {index->
                val deadline=SystemClock.elapsedRealtime()+20000
                while(state(activity).optInt("frameIndex",-1)<15 && SystemClock.elapsedRealtime()<deadline) SystemClock.sleep(100)
                assertTrue(state(activity).getInt("frameIndex")>=15)
                val runningPss=android.os.Debug.getPss();click("停止感知")
                val end=SystemClock.elapsedRealtime()+10000
                while((state(activity).getBoolean("releasePending") || state(activity).getBoolean("spatialReleasePending")) && SystemClock.elapsedRealtime()<end) SystemClock.sleep(100)
                assertFalse(state(activity).getBoolean("releasePending"));assertFalse(state(activity).getBoolean("spatialReleasePending"))
                assertFalse(state(activity).getBoolean("running"));assertFalse(state(activity).getBoolean("modelReady"))
                cycles.put(JSONObject().put("cycle",index).put("running_pss_kb",runningPss).put("released_pss_kb",android.os.Debug.getPss()))
                if(index<2) click("开始感知")
            }
        } finally {
            java.io.File(context.filesDir,"lifecycle-memory-validation.json").writeText(JSONObject().put("schema_version",1).put("cycles",cycles).toString(2))
            instrumentation.runOnMainSync {activity.finish()};instrumentation.waitForIdleSync()
        }
    }
    private val instrumentation get()=InstrumentationRegistry.getInstrumentation()
    private fun state(activity: MainActivity): JSONObject {var result=JSONObject();instrumentation.runOnMainSync {result=activity.sessionDiagnostics()};return result}
    private fun nodes(root: AccessibilityNodeInfo): List<AccessibilityNodeInfo> = listOf(root)+(0 until root.childCount).flatMap {i->root.getChild(i)?.let {nodes(it)} ?: emptyList()}
    private fun click(label: String) {
        val deadline=SystemClock.elapsedRealtime()+5000
        while(SystemClock.elapsedRealtime()<deadline) {
            val root=instrumentation.uiAutomation.rootInActiveWindow
            var node=root?.let {nodes(it).firstOrNull {n->n.text?.toString()==label || n.contentDescription?.toString()==label}}
            while(node!=null && !node.isClickable) node=node.parent
            if(node?.performAction(AccessibilityNodeInfo.ACTION_CLICK)==true) {instrumentation.waitForIdleSync();return}
            SystemClock.sleep(100)
        }
        fail("Accessible action missing: $label")
    }
    @Test fun ordinaryModeHasLargeActionsAndNoResearchControlsAndStopPersists() {
        val context=instrumentation.targetContext
        context.getSharedPreferences("perception",0).edit().putBoolean("auto_start",true).commit()
        android.os.ParcelFileDescriptor.AutoCloseInputStream(instrumentation.uiAutomation.executeShellCommand("pm grant ${context.packageName} ${Manifest.permission.CAMERA}")).use {it.readBytes()}
        val activity=instrumentation.startActivitySync(Intent(context,MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) as MainActivity
        var recreated: MainActivity?=null
        try {
            val deadline=SystemClock.elapsedRealtime()+20000
            while(state(activity).optInt("frameIndex",-1)<1 && SystemClock.elapsedRealtime()<deadline) SystemClock.sleep(100)
            assertTrue(state(activity).getBoolean("semanticEnabled"));assertTrue(state(activity).getInt("frameIndex")>=1)
            val root=instrumentation.uiAutomation.rootInActiveWindow
            val all=nodes(root);val texts=all.mapNotNull {it.text?.toString()}
            assertFalse(texts.any {it.contains("SegFormer") || it.contains("Spatial Grid") || it.contains("透明度")})
            for(label in listOf("描述周围","静音提醒","停止感知","设置")) {
                var node: AccessibilityNodeInfo?=all.firstOrNull {it.text?.toString()==label}
                while(node!=null && !node.isClickable) node=node.parent
                assertNotNull(label,node)
                val bounds=Rect();node!!.getBoundsInScreen(bounds)
                val minimum=48*context.resources.displayMetrics.density
                assertTrue("$label touch target $bounds",bounds.width()>=minimum && bounds.height()>=minimum)
            }
            click("静音提醒");assertTrue(state(activity).getBoolean("muted"))
            click("恢复提醒");assertFalse(state(activity).getBoolean("muted"))
            click("描述周围")
            click("停止感知");assertFalse(state(activity).getBoolean("running"));assertTrue(state(activity).getBoolean("explicitStop"))
            click("设置");assertTrue(state(activity).getBoolean("settingsPanel"))
            click("开发者工具");assertTrue(state(activity).getBoolean("developerPanel"))
            click("返回画面");SystemClock.sleep(700);assertFalse(state(activity).getBoolean("running"))
            instrumentation.runOnMainSync {activity.recreate()};instrumentation.waitForIdleSync()
            instrumentation.runOnMainSync {recreated=androidx.test.runner.lifecycle.ActivityLifecycleMonitorRegistry.getInstance().getActivitiesInStage(androidx.test.runner.lifecycle.Stage.RESUMED).filterIsInstance<MainActivity>().firstOrNull()}
            assertNotNull("Recreated activity must be inspected",recreated)
            assertFalse(state(recreated!!).getBoolean("running"));assertTrue(state(recreated!!).getBoolean("explicitStop"))
        } finally {instrumentation.runOnMainSync {recreated?.finish();activity.finish()};instrumentation.waitForIdleSync()}
    }
}
