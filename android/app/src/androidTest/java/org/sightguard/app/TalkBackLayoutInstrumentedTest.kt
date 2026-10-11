package org.sightguard.app

import android.app.UiAutomation
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Rect
import android.os.ParcelFileDescriptor
import android.os.SystemClock
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityManager
import android.view.accessibility.AccessibilityNodeInfo
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.util.concurrent.CopyOnWriteArrayList

/** Requires the installed TalkBack service to be enabled by the device-validation command. */
class TalkBackLayoutInstrumentedTest {
    @Test fun actualTalkBackFocusLargeFontRotationAndAudioCoexistence() {
        val ins=InstrumentationRegistry.getInstrumentation();val context=ins.targetContext
        val automation=ins.getUiAutomation(UiAutomation.FLAG_DONT_SUPPRESS_ACCESSIBILITY_SERVICES)
        fun shell(command: String)=ParcelFileDescriptor.AutoCloseInputStream(automation.executeShellCommand(command)).bufferedReader().use {it.readText().trim()}
        fun nodes(n: AccessibilityNodeInfo): List<AccessibilityNodeInfo> = listOf(n)+(0 until n.childCount).flatMap {n.getChild(it)?.let {c->nodes(c)} ?: emptyList()}
        val manager=context.getSystemService(AccessibilityManager::class.java)
        assertTrue("Enable actual TalkBack before this test",manager.isTouchExplorationEnabled)
        val font=shell("settings get system font_scale");val rotation=shell("settings get system user_rotation");val auto=shell("settings get system accelerometer_rotation")
        val events=CopyOnWriteArrayList<String>();val playback=CopyOnWriteArrayList<SpeechPlaybackEvent>();val checks=JSONArray()
        automation.setOnAccessibilityEventListener {e->if(e.packageName==context.packageName && e.eventType in setOf(AccessibilityEvent.TYPE_VIEW_ACCESSIBILITY_FOCUSED,AccessibilityEvent.TYPE_ANNOUNCEMENT)) events.add("${e.eventType}:${e.text}:${e.contentDescription}")}
        var activity: MainActivity?=null;var speech: SpeechController?=null
        fun state(): JSONObject {var value=JSONObject();ins.runOnMainSync {value=activity!!.sessionDiagnostics()};return value}
        fun screenshot(name: String) {automation.takeScreenshot().let {b->File(context.filesDir,name).outputStream().use {b.compress(Bitmap.CompressFormat.PNG,100,it)};b.recycle()}}
        fun checkActions(mode: String) {
            repeat(8) {automation.rootInActiveWindow?.let {root->nodes(root).firstOrNull {it.isScrollable}?.performAction(AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD)}}
            SystemClock.sleep(250)
            for(label in listOf("描述周围","静音提醒","停止感知","设置")) {
                var found: AccessibilityNodeInfo?=null
                for(attempt in 0..8) {
                    val all=automation.rootInActiveWindow?.let {nodes(it)} ?: emptyList()
                    var n=all.firstOrNull {it.text?.toString()==label}
                    while(n!=null && !n.isClickable) n=n.parent
                    val r=Rect();n?.getBoundsInScreen(r)
                    if(n!=null && n.isVisibleToUser && r.height()>=48*context.resources.displayMetrics.density) {found=n;break}
                    all.firstOrNull {it.isScrollable}?.performAction(AccessibilityNodeInfo.ACTION_SCROLL_FORWARD);SystemClock.sleep(250)
                }
                assertNotNull("$mode accessible action $label",found)
                val bounds=Rect();found!!.getBoundsInScreen(bounds)
                assertTrue(bounds.width()>=48*context.resources.displayMetrics.density)
                assertTrue(found.performAction(AccessibilityNodeInfo.ACTION_ACCESSIBILITY_FOCUS))
                checks.put(JSONObject().put("mode",mode).put("label",label).put("bounds",bounds.toShortString()).put("focus",found.isAccessibilityFocused))
                SystemClock.sleep(400)
            }
        }
        try {
            shell("settings put system font_scale 2.0");shell("settings put system accelerometer_rotation 0");shell("settings put system user_rotation 0")
            context.getSharedPreferences("perception",0).edit().putBoolean("auto_start",true).putBoolean("muted",false).commit()
            activity=ins.startActivitySync(Intent(context,MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) as MainActivity
            val deadline=SystemClock.elapsedRealtime()+20000
            while(state().optInt("frameIndex",-1)<1 && SystemClock.elapsedRealtime()<deadline) SystemClock.sleep(100)
            assertTrue(state().getInt("frameIndex")>=1)
            screenshot("ui-font200-portrait.png");checkActions("font200_portrait")
            val focusCount=events.size;SystemClock.sleep(3000)
            assertEquals("Automatic updates must not steal accessibility focus",focusCount,events.size)
            shell("settings put system user_rotation 1");SystemClock.sleep(3000)
            screenshot("ui-font200-landscape.png");checkActions("font200_landscape")
            ins.runOnMainSync {speech=SpeechController(context,{}, {}, {playback.add(it)});speech!!.startLocalFallbackForTest()}
            SystemClock.sleep(1800)
            ins.runOnMainSync {speech!!.offer("前方检测到墙面。",false,"semantic:talkback-validation",SystemClock.elapsedRealtime()+2500,Direction.CENTER)}
            val end=SystemClock.elapsedRealtime()+4000
            while(playback.none {it.phase==SpeechPlaybackPhase.COMPLETED} && SystemClock.elapsedRealtime()<end) SystemClock.sleep(50)
            assertEquals(1,playback.count {it.phase==SpeechPlaybackPhase.ON_START})
            assertTrue(manager.isTouchExplorationEnabled)
            shell("settings put system user_rotation 0");shell("wm size 420x900");SystemClock.sleep(2500)
            screenshot("ui-font200-narrow.png");checkActions("font200_narrow")
        } finally {
            File(context.filesDir,"talkback-layout-validation.json").writeText(JSONObject().put("schema_version",1).put("service_active",manager.isTouchExplorationEnabled)
                .put("checks",checks).put("accessibility_focus_and_announcement_events",JSONArray(events)).put("audio_phases",JSONArray(playback.map {it.phase.logName}))
                .put("limitation","Platform focus and app audio start verified; human Chinese pronunciation and physical-device interaction not certified").toString(2))
            ins.runOnMainSync {speech?.close();activity?.finish()};ins.waitForIdleSync()
            shell("wm size reset");shell("settings put system font_scale $font");shell("settings put system user_rotation $rotation");shell("settings put system accelerometer_rotation $auto")
            automation.setOnAccessibilityEventListener(null)
        }
    }
}
