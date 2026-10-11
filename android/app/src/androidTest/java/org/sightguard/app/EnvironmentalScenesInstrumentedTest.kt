package org.sightguard.app

import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Color
import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.CopyOnWriteArrayList

/** Actual Android ONNX observations of real scenes. Misses are exported, never synthesized away. */
class EnvironmentalScenesInstrumentedTest {
    @Test fun realSceneMasksEventsAndActualAudioAreRecordedIncludingFailures() {
        val ins=InstrumentationRegistry.getInstrumentation();val context=ins.targetContext
        val manifest=JSONObject(ins.context.assets.open("awareness-scenes.json").bufferedReader().use {it.readText()})
        val cases=manifest.getJSONArray("cases");val results=JSONArray();val events=CopyOnWriteArrayList<SpeechPlaybackEvent>()
        val engine=SemanticEnvironmentEngine(loadEnvironmentConfig(context));lateinit var speech: SpeechController
        ins.runOnMainSync {speech=SpeechController(context,{}, {}, {e->events.add(e);if(e.phase==SpeechPlaybackPhase.ON_START) engine.playbackStarted(e.eventId!!)});speech.startLocalFallbackForTest()}
        val recognized=mutableSetOf<String>();val played=mutableSetOf<String>()
        try {
            SpatialDetector(context).use {detector->
                for(index in 0 until cases.length()) {
                    val case=cases.getJSONObject(index);val name=case.getString("file")
                    val bytes=ins.context.assets.open("awareness/$name").use {it.readBytes()}
                    val hash=MessageDigest.getInstance("SHA-256").digest(bytes).joinToString(""){"%02x".format(it)}
                    assertEquals(case.getString("sha256"),hash)
                    val bitmap=BitmapFactory.decodeByteArray(bytes,0,bytes.size);assertNotNull(bitmap)
                    ins.runOnMainSync {speech.cancelAlerts("fixture_scene_changed");engine.reset()}
                    val observations=JSONArray();val caseIds=mutableSetOf<String>()
                    for(i in 0..3) {
                        val start=SystemClock.elapsedRealtime();val key=SpatialFrameKey(1000L+index,0,i,start.toDouble(),bitmap.width,bitmap.height,start)
                        val evidence=detector.analyzeAll(bitmap,key)
                        val parts=SemanticStructureMeasure(engine.config).components(evidence.cells)
                        var notices=emptyList<EnvironmentNotice>()
                        ins.runOnMainSync {
                            notices=engine.observe(evidence,SystemClock.elapsedRealtime(),true,false)
                            val wanted=case.optString("preferred_kind")
                            val notice=notices.firstOrNull {it.kind.name==wanted} ?: notices.firstOrNull {it.kind.name!="STEP"}
                            if(notice!=null) {caseIds.add(notice.id);speech.offer(notice.text,false,notice.id,notice.validUntilMs,notice.direction)}
                        }
                        parts.filter {it.supported}.forEach {recognized.add(it.kind.name)}
                        observations.put(JSONObject().put("frame",i).put("source_age_ms",SystemClock.elapsedRealtime()-start).put("latency_ms",evidence.latencyMs)
                            .put("overview",SemanticEvidence.summarize(evidence.cells,emptyList(),SemanticSourceQuality()).overview)
                            .put("class_counts",JSONObject(evidence.cells.groupingBy{it.classId.toString()}.eachCount()))
                            .put("structures",JSONArray(parts.map {environmentSupportJson(it)})).put("notices",JSONArray(notices.map {it.id})))
                        if(i==0) {
                            val overlay=bitmap.copy(android.graphics.Bitmap.Config.ARGB_8888,true);val canvas=Canvas(overlay)
                            val paint=Paint().apply {color=Color.argb(110,255,60,170)}
                            val ids=case.getJSONArray("target_class_ids");val targets=(0 until ids.length()).map {ids.getInt(it)}.toSet()
                            evidence.cells.filter {it.classId in targets}.forEach {c->canvas.drawRect(c.left*bitmap.width,c.top*bitmap.height,c.right*bitmap.width,c.bottom*bitmap.height,paint)}
                            File(context.filesDir,"awareness-mask-$index.png").outputStream().use {overlay.compress(android.graphics.Bitmap.CompressFormat.PNG,100,it)};overlay.recycle()
                        }
                        SystemClock.sleep((650-(SystemClock.elapsedRealtime()-start)).coerceAtLeast(0))
                    }
                    val deadline=SystemClock.elapsedRealtime()+3500
                    while(caseIds.isNotEmpty() && events.none {it.eventId in caseIds && it.phase==SpeechPlaybackPhase.COMPLETED} && SystemClock.elapsedRealtime()<deadline) SystemClock.sleep(50)
                    val started=events.filter {it.eventId in caseIds && it.phase==SpeechPlaybackPhase.ON_START}
                    started.forEach {e->assertTrue("stale audio start",e.elapsedRealtimeMs<e.validUntilElapsedMs);assertFalse(SpeechPromptCatalog.resolve(e.spokenText!!).usedGeneric);played.add(e.requestedText)}
                    assertEquals("persistent fixture must not repeat",started.map {it.eventId}.toSet().size,started.size)
                    results.put(JSONObject().put("case",case).put("observations",observations).put("audio_started",JSONArray(started.map {it.spokenText})))
                    bitmap.recycle()
                }
            }
            assertTrue("Need real stair structure",recognized.contains("STAIRS"))
            assertTrue("Need real column/pole structure",recognized.contains("POLE"))
            assertTrue("Need real stair Chinese audio",played.any {it.contains("楼梯")})
            assertTrue("Need real column Chinese audio",played.any {it.contains("杆状结构")})
        } finally {
            val report=JSONObject().put("schema_version",1).put("source","actual SegFormer Android CPU inference of real photos; repeated stills, not CameraX motion acceptance")
                .put("model_sha256",SpatialDetector.MODEL_SHA).put("cases",results).put("automatic_wall_proximity",false)
                .put("audio_events",JSONArray(events.map {e->JSONObject().put("phase",e.phase.logName).put("event",e.eventId).put("requested",e.requestedText).put("spoken",e.spokenText).put("resource",e.resourceName).put("elapsed_ms",e.elapsedRealtimeMs).put("deadline_ms",e.validUntilElapsedMs).put("detail",e.detail)}))
            File(context.filesDir,"awareness-scenes-actual.json").writeText(report.toString(2))
            ins.runOnMainSync {speech.close()}
        }
    }
}
