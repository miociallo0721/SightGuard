package org.sightguard.app

import android.graphics.BitmapFactory
import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/** Public static photographs are not a moving-camera or passage validation. */
class LocalSpaceInstrumentedTest {
    private val instrumentation=InstrumentationRegistry.getInstrumentation()
    private val context=instrumentation.targetContext
    private fun detections(j: JSONObject): List<Detection> = j.getJSONArray("detections").let {a -> (0 until a.length()).map {i ->
        val x=a.getJSONArray(i);Detection(x.getInt(0),x.getDouble(1).toFloat(),x.getDouble(2).toFloat(),x.getDouble(3).toFloat(),x.getDouble(4).toFloat(),x.getDouble(5).toFloat())}}
    private fun grid(s: GroundEvidence,d: List<Detection>): SpatialGridEvidence = SpatialGridFusion(GridConfig(minConfidence=.65f,includeFurniture=true)).fuse(
        s.key,GridSemanticSource(s.key,s.cells,s.modelSha),null,GridDetectionSource(s.key,d,"public-replay-or-actual-yoloe"),MotionEvidence(MotionQuality.UNKNOWN,"static photograph, no device-motion correspondence"))
    @Test fun replayEveryPublicModelOutputAndPreserveFailures() {
        val index=JSONArray(instrumentation.context.assets.open("local_space/index.json").bufferedReader().use {it.readText()})
        val results=JSONArray()
        for(i in 0 until index.length()) {
            val item=index.getJSONObject(i)
            val j=JSONObject(instrumentation.context.assets.open("local_space/"+item.getString("image").replace(".jpg",".json")).bufferedReader().use {it.readText()})
            val cells=j.getJSONArray("cells").let {a-> (0 until a.length()).map {n->val c=a.getJSONArray(n);SemanticCell(c.getDouble(0).toFloat(),c.getDouble(1).toFloat(),c.getDouble(2).toFloat(),c.getDouble(3).toFloat(),c.getInt(4),c.getDouble(5).toFloat())}}
            val base=SpatialFrameKey(1,0,0,0.0,j.getInt("width"),j.getInt("height"),0)
            val sem=GroundEvidence(base,cells,j.getDouble("semantic_ms").toLong(),0,"PC actual ONNX",j.getString("model_sha256"))
            val fused=grid(sem,detections(j));val engine=LocalSpaceCandidateEvaluator(loadLocalSpaceConfig(context))
            var r: LocalSpaceRecord?=null
            for(n in 1..3) {
                val key=base.copy(frameIndex=n,timestampMs=n*800.0,capturedAtMs=n*800L)
                r=engine.observe(fused.copy(key=key),sem.copy(key=key),n*800L+30,true)
            }
            val report=localSpaceJson(r!!).put("image",j.getString("image")).put("gt_bands",j.getJSONArray("gt_bands"))
                .put("timing_scope","cached output replay, synthetic 800ms observation intervals; not live timing")
            if(r.direction!=null && r.status=="SUPPORTED_CANDIDATE") {
                val col=when(r.direction) {Direction.LEFT->0;Direction.CENTER->1;else->2}
                val gt=j.getJSONArray("gt_bands").getJSONObject(col)
                report.put("annotation_support",gt.getDouble("ground_fraction")>=.65 && gt.getDouble("hazard_fraction")<.01)
            }
            results.put(report)
        }
        File(context.filesDir,"local_space_public_replay.json").writeText(JSONObject().put("cases",results).put("count",index.length()).toString(2))
        assertEquals(88,index.length())
    }
    @Test fun actualOnnxPhotographsAndAudio() {
        val selected=JSONArray(instrumentation.context.assets.open("local_space/actual_cases.json").bufferedReader().use {it.readText()})
        val semanticSize=InstrumentationRegistry.getArguments().getString("local_space_size","256").toInt()
        val records=JSONArray();val events=CopyOnWriteArrayList<SpeechPlaybackEvent>()
        lateinit var speech: SpeechController
        instrumentation.runOnMainSync {speech=SpeechController(context,{}, {}, {events+=it});speech.startLocalFallbackForTest()}
        try {
            Detector(context,true).use {yolo -> SpatialDetector(context,1,semanticSize).use {semantic ->
                for(i in 0 until selected.length()) {
                    val name=selected.getString(i)
                    val bitmap=instrumentation.context.assets.open("local_space/$name").use {BitmapFactory.decodeStream(it)}
                    val engine=LocalSpaceCandidateEvaluator(loadLocalSpaceConfig(context));val observations=JSONArray()
                    for(n in 1..3) {
                        val now=SystemClock.elapsedRealtime()
                        val key=SpatialFrameKey(i.toLong()+1,0,n,now.toDouble(),bitmap.width,bitmap.height,now)
                        val d=yolo.detect(bitmap);val sem=semantic.analyzeAll(bitmap,key)
                        val gridStart=SystemClock.elapsedRealtime()
                        val g=grid(sem,d)
                        val r=engine.observe(g,sem,SystemClock.elapsedRealtime(),true)
                        val localMs=SystemClock.elapsedRealtime()-gridStart
                        observations.put(localSpaceJson(r).put("source_age_ms",SystemClock.elapsedRealtime()-now).put("semantic_ms",sem.latencyMs).put("local_ms",localMs).put("pss_kb",android.os.Debug.getPss()))
                        val notice=engine.pending(SystemClock.elapsedRealtime())
                        if(notice!=null) instrumentation.runOnMainSync {
                            speech.offer(notice.text!!,false,notice.eventId,notice.validUntilMs,notice.direction)
                        }
                        SystemClock.sleep(200)
                    }
                    SystemClock.sleep(5000)
                    records.put(JSONObject().put("image",name).put("observations",observations))
                    bitmap.recycle()
                }
            }}
        } finally {
            File(context.filesDir,"local_space_actual.json").writeText(JSONObject().put("records",records).put("device",android.os.Build.MODEL)
                .put("semantic_size",semanticSize).put("scope","actual Android ONNX; repeated static public photos, not CameraX motion")
                .put("audio",JSONArray(events.map {JSONObject().put("phase",it.phase.logName).put("text",it.spokenText).put("detail",it.detail).put("resource",it.resourceName).put("event_id",it.eventId).put("elapsed_ms",it.elapsedRealtimeMs).put("deadline_ms",it.validUntilElapsedMs)})).toString(2))
            instrumentation.runOnMainSync {speech.close()}
        }
        assertEquals(selected.length(),records.length())
        assertTrue("Actual public-model candidate must reach onStart",events.any {it.phase==SpeechPlaybackPhase.ON_START})
        assertTrue(events.filter {it.phase==SpeechPlaybackPhase.ON_START}.all {it.elapsedRealtimeMs<it.validUntilElapsedMs})
    }
    @Test fun exactChineseClipsStartAndObsoleteCandidateCannotPlay() {
        val events=CopyOnWriteArrayList<SpeechPlaybackEvent>();val starts=CountDownLatch(3)
        lateinit var speech: SpeechController
        instrumentation.runOnMainSync {speech=SpeechController(context,{}, {}, {e->events+=e;if(e.phase==SpeechPlaybackPhase.ON_START) starts.countDown()});speech.startLocalFallbackForTest()}
        try {
            for(direction in Direction.entries) {
                val text=LocalSpaceRecord(SpatialFrameKey(1,0,1,1.0,300,300,0),"SUPPORTED_CANDIDATE","synthetic audio test",direction=direction).text!!
                instrumentation.runOnMainSync {assertTrue(speech.offer(text,false,"local-space:audio:${direction}",SystemClock.elapsedRealtime()+1200,direction))}
                SystemClock.sleep(6000)
            }
            assertTrue(starts.await(1,TimeUnit.SECONDS))
            assertEquals(3,events.count {it.phase==SpeechPlaybackPhase.ON_START && it.resourceName?.startsWith("ar_corridor_")==true})
            instrumentation.runOnMainSync {
                speech.offer("前方有行人",true,"risk",SystemClock.elapsedRealtime()+2000)
                speech.offer("画面左侧可见地面较连续，通行条件尚未确认。",false,"local-space:obsolete",SystemClock.elapsedRealtime()+1000,Direction.LEFT)
                speech.cancelLocalSpaceExcept(emptySet(),"obstacle_entered")
            }
            SystemClock.sleep(3000)
            assertFalse(events.any {it.eventId=="local-space:obsolete" && it.phase==SpeechPlaybackPhase.ON_START})
            assertTrue(events.any {it.eventId=="risk" && it.phase==SpeechPlaybackPhase.ON_START})
            instrumentation.runOnMainSync {
                speech.offer("画面左侧可见地面较连续，通行条件尚未确认。",false,"local-space:revoked",SystemClock.elapsedRealtime()+1200,Direction.LEFT)
            }
            SystemClock.sleep(500)
            assertTrue(events.any {it.eventId=="local-space:revoked" && it.phase==SpeechPlaybackPhase.ON_START})
            instrumentation.runOnMainSync {speech.cancelLocalSpaceExcept(emptySet(),"candidate_changed")}
            assertTrue(events.any {it.eventId=="local-space:revoked" && it.phase==SpeechPlaybackPhase.CANCELLED})
        } finally {
            File(context.filesDir,"local_space_audio.json").writeText(JSONArray(events.map {JSONObject().put("phase",it.phase.logName).put("text",it.spokenText).put("event_id",it.eventId).put("detail",it.detail)}).toString(2))
            instrumentation.runOnMainSync {speech.close()}
        }
    }
}
