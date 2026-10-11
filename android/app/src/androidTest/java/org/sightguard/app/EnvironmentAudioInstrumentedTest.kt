package org.sightguard.app

import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.util.concurrent.CopyOnWriteArrayList

/** Resource playback checks; these are deliberately not model-recognition acceptance claims. */
class EnvironmentAudioInstrumentedTest {
    @Test fun everyExactDirectionalEnvironmentClipActuallyStartsAndCompletes() {
        val ins=InstrumentationRegistry.getInstrumentation();val events=CopyOnWriteArrayList<SpeechPlaybackEvent>()
        lateinit var speech: SpeechController
        ins.runOnMainSync {speech=SpeechController(ins.targetContext,{}, {}, {events.add(it)});speech.startLocalFallbackForTest()}
        try {
            for(kind in EnvironmentKind.entries) for(direction in Direction.entries) {
                val id="semantic:resource-validation:${kind.name}:${direction.name}";val text=environmentText(kind,direction)
                ins.runOnMainSync {assertTrue(speech.offer(text,false,id,SystemClock.elapsedRealtime()+2500,direction))}
                val deadline=SystemClock.elapsedRealtime()+5000
                while(events.none {it.eventId==id && it.phase==SpeechPlaybackPhase.COMPLETED} && SystemClock.elapsedRealtime()<deadline) SystemClock.sleep(30)
                val started=events.filter {it.eventId==id && it.phase==SpeechPlaybackPhase.ON_START}
                assertEquals(id,1,started.size);assertEquals(text,started.single().spokenText)
                assertTrue(started.single().elapsedRealtimeMs<started.single().validUntilElapsedMs)
                assertFalse(SpeechPromptCatalog.resolve(text).usedGeneric)
                assertTrue(events.any {it.eventId==id && it.phase==SpeechPlaybackPhase.COMPLETED})
            }
        } finally {
            File(ins.targetContext.filesDir,"environment-audio-validation.json").writeText(JSONObject().put("schema_version",1)
                .put("source","explicit resource playback only; STEP automatic alerts remain disabled")
                .put("events",JSONArray(events.map {e->JSONObject().put("event",e.eventId).put("phase",e.phase.logName).put("text",e.spokenText).put("resource",e.resourceName).put("elapsed_ms",e.elapsedRealtimeMs).put("deadline_ms",e.validUntilElapsedMs)})).toString(2))
            ins.runOnMainSync {speech.close()}
        }
    }
}
