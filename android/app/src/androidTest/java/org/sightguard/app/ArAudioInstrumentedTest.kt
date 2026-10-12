package org.sightguard.app

import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.CopyOnWriteArrayList

/** Actual playback of experimental wording. This is NOT recognition/metric calibration evidence. */
class ArAudioInstrumentedTest {
    @Test fun experimentalChinesePromptsActuallyStartAndComplete() {
        val ins=InstrumentationRegistry.getInstrumentation();val events=CopyOnWriteArrayList<SpeechPlaybackEvent>()
        val preferences=ins.targetContext.getSharedPreferences("perception",0)
        val autoStart=preferences.getBoolean("auto_start",true)
        preferences.edit().putBoolean("auto_start",false).commit()
        ins.runOnMainSync {
            androidx.test.runner.lifecycle.ActivityLifecycleMonitorRegistry.getInstance()
                .getActivitiesInStage(androidx.test.runner.lifecycle.Stage.RESUMED).toList().forEach {it.finish()}
        }
        ins.waitForIdleSync()
        lateinit var speech: SpeechController
        ins.runOnMainSync {speech=SpeechController(ins.targetContext,{}, {}, {events.add(it)});speech.startLocalFallbackForTest()}
        try {
            for((i,text) in listOf("前方墙面可能较近，请注意。","相机可能正在靠近前方墙面，请注意。","画面左侧可见地面较连续，通行条件尚未确认。","画面中央可见地面较连续，通行条件尚未确认。","画面右侧可见地面较连续，通行条件尚未确认。").withIndex()) {
                val id="research:resource-test:$i"
                ins.runOnMainSync {assertTrue(speech.offer(text,false,id,SystemClock.elapsedRealtime()+500))}
                val deadline=SystemClock.elapsedRealtime()+10000
                while(events.none {it.eventId==id && it.phase==SpeechPlaybackPhase.COMPLETED} && SystemClock.elapsedRealtime()<deadline) SystemClock.sleep(30)
                val start=events.single {it.eventId==id && it.phase==SpeechPlaybackPhase.ON_START}
                assertEquals(text,start.spokenText);assertTrue(start.elapsedRealtimeMs<start.validUntilElapsedMs)
                assertTrue(events.any {it.eventId==id && it.phase==SpeechPlaybackPhase.COMPLETED})
            }
            ins.runOnMainSync {
                speech.setMuted(true)
                assertFalse(speech.offer("前方墙面可能较近，请注意。",false,"research:muted",SystemClock.elapsedRealtime()+500))
                speech.setMuted(false)
                assertFalse(speech.offer("前方墙面可能较近，请注意。",false,"research:expired",SystemClock.elapsedRealtime()-1))
            }
        } finally {
            ins.runOnMainSync {speech.close()};ins.waitForIdleSync()
            preferences.edit().putBoolean("auto_start",autoStart).commit()
            File(ins.targetContext.filesDir,"ar-audio-validation.json").writeText(JSONObject().put("scope","resource_playback_only_not_physical_scene_acceptance")
                .put("events",JSONArray(events.map {JSONObject().put("id",it.eventId).put("phase",it.phase.logName).put("text",it.spokenText).put("resource",it.resourceName).put("elapsed_ms",it.elapsedRealtimeMs).put("deadline_ms",it.validUntilElapsedMs)})).toString(2))
        }
    }
}
