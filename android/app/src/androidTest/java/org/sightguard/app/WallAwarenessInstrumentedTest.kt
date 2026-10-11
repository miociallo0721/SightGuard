package org.sightguard.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/** Real photographs -> Android ORT -> presence state machine -> actual MediaPlayer start. */
class WallAwarenessInstrumentedTest {
    @Test fun actualWallMaskPresencePlaybackPersistenceAndAbsence() {
        val instrumentation=InstrumentationRegistry.getInstrumentation()
        val context=instrumentation.targetContext
        val config=loadEnvironmentConfig(context)
        val engine=SemanticEnvironmentEngine(config)
        val events=CopyOnWriteArrayList<SpeechPlaybackEvent>()
        val started=CountDownLatch(1)
        val completed=CountDownLatch(1)
        val observations=JSONArray()
        val bytes=instrumentation.context.assets.open("v03-room.jpg").use { it.readBytes() }
        val wall=BitmapFactory.decodeByteArray(bytes,0,bytes.size)
        val outdoor=instrumentation.context.assets.open("v03-outdoor.jpg").use { BitmapFactory.decodeStream(it) }
        lateinit var speech: SpeechController
        instrumentation.runOnMainSync {
            speech=SpeechController(context,{}, {}, { event ->
                events+=event
                if(event.phase==SpeechPlaybackPhase.ON_START) { engine.playbackStarted(event.eventId!!); started.countDown() }
                if(event.phase==SpeechPlaybackPhase.COMPLETED) completed.countDown()
            })
            speech.startLocalFallbackForTest()
        }
        try {
            SpatialDetector(context).use { detector ->
                for(i in 0..7) {
                    val now=SystemClock.elapsedRealtime()
                    val key=SpatialFrameKey(100,0,i,now.toDouble(),wall.width,wall.height,now)
                    val evidence=detector.analyzeAll(wall,key)
                    assertTrue("real wall class 0 missing",evidence.cells.count{it.classId==0}>10)
                    var notices: List<EnvironmentNotice> = emptyList()
                    instrumentation.runOnMainSync {
                        notices=engine.observe(evidence,SystemClock.elapsedRealtime(),true,false)
                        if(i==0) assertTrue(notices.isEmpty())
                        notices.forEach { notice -> assertTrue(speech.offer(notice.kind.text,false,notice.id,notice.validUntilMs,Direction.CENTER)) }
                    }
                    observations.put(JSONObject().put("frame",i).put("source_ms",now).put("latency_ms",evidence.latencyMs)
                        .put("completion_age_ms",evidence.completedAtMs-now).put("status",engine.status).put("wall_cells",evidence.cells.count{it.classId==0})
                        .put("class_counts",JSONObject(evidence.cells.groupingBy{it.classId.toString()}.eachCount()))
                        .put("wall_roi_fraction",engine.measure(evidence.cells).first().roiFraction).put("wall_confidence",engine.measure(evidence.cells).first().confidence)
                        .put("notices",JSONArray(notices.map{it.id})).put("depth_available",false).put("approach_enabled",engine.approachEnabled))
                    val grid=SpatialGridFusion().fuse(key,GridSemanticSource(key,evidence.cells,SpatialDetector.MODEL_SHA),null,
                        GridDetectionSource(key,emptyList(),"no detector required for presence"),MotionEvidence(MotionQuality.UNKNOWN,"fixture has no IMU"))
                    assertTrue(grid.cells.any { it.state=="WALL" })
                    if(i==0) {
                        val overlay=wall.copy(Bitmap.Config.ARGB_8888,true); val canvas=Canvas(overlay)
                        val paint=Paint().apply{color=Color.argb(100,255,0,130)}
                        evidence.cells.filter{it.classId==0}.forEach { cell -> canvas.drawRect(cell.left*wall.width,cell.top*wall.height,cell.right*wall.width,cell.bottom*wall.height,paint) }
                        File(context.filesDir,"wall_mask_actual.png").outputStream().use{overlay.compress(Bitmap.CompressFormat.PNG,100,it)}; overlay.recycle()
                    }
                    SystemClock.sleep((config.sampleIntervalMs-(SystemClock.elapsedRealtime()-now)).coerceAtLeast(0))
                }
                assertTrue("No actual Chinese audio onStart",started.await(5,TimeUnit.SECONDS))
                assertTrue("No audio completion",completed.await(5,TimeUnit.SECONDS))
                assertEquals(1,events.count{it.phase==SpeechPlaybackPhase.ON_START})
                val actual=events.single{it.phase==SpeechPlaybackPhase.ON_START}
                assertEquals("前方检测到墙面。",actual.spokenText)
                assertEquals("environment_wall",actual.resourceName)
                assertEquals("local_media_player",actual.detail)
                assertTrue(actual.elapsedRealtimeMs<actual.validUntilElapsedMs)
                val until=SystemClock.elapsedRealtime()+config.recoveryMs+1000
                var i=8
                while(SystemClock.elapsedRealtime()<until) {
                    val now=SystemClock.elapsedRealtime()
                    val evidence=detector.analyzeAll(outdoor,SpatialFrameKey(100,0,i++,now.toDouble(),outdoor.width,outdoor.height,now))
                    assertFalse("negative photograph unexpectedly supports wall",engine.measure(evidence.cells).first().supported)
                    instrumentation.runOnMainSync { assertTrue(engine.observe(evidence,SystemClock.elapsedRealtime(),true,false).isEmpty()) }
                    SystemClock.sleep(config.sampleIntervalMs)
                }
                instrumentation.runOnMainSync { assertTrue(engine.validIds(SystemClock.elapsedRealtime()).isEmpty()) }
            }
        } finally {
            val report=JSONObject().put("schema_version",1).put("device",android.os.Build.MODEL).put("api",android.os.Build.VERSION.SDK_INT)
                .put("model_sha256",SpatialDetector.MODEL_SHA).put("input_sha256",MessageDigest.getInstance("SHA-256").digest(bytes).joinToString(""){"%02x".format(it)})
                .put("source","v03-room.jpg; original CC0 photograph; repeated static input, not moving camera validation")
                .put("observations",observations).put("audio_events",JSONArray(events.map{ e -> JSONObject().put("phase",e.phase.logName).put("event",e.eventId).put("text",e.spokenText).put("elapsed_ms",e.elapsedRealtimeMs).put("deadline_ms",e.validUntilElapsedMs).put("resource",e.resourceName).put("detail",e.detail) }))
            File(context.filesDir,"wall_awareness_actual.json").writeText(report.toString(2))
            instrumentation.runOnMainSync{speech.close()}; wall.recycle(); outdoor.recycle()
        }
    }

    @Test fun staleSemanticRequestIsCancelledWithoutCancellingRiskSpeech() {
        val instrumentation=InstrumentationRegistry.getInstrumentation()
        val events=CopyOnWriteArrayList<SpeechPlaybackEvent>()
        val start=CountDownLatch(1)
        lateinit var speech: SpeechController
        instrumentation.runOnMainSync {
            speech=SpeechController(instrumentation.targetContext,{}, {}, { e -> events+=e; if(e.phase==SpeechPlaybackPhase.ON_START && e.eventId=="risk") start.countDown() })
            speech.startLocalFallbackForTest()
            speech.offer("前方有行人",true,"risk",SystemClock.elapsedRealtime()+2000)
            speech.offer(EnvironmentKind.WALL.text,false,"semantic:obsolete",SystemClock.elapsedRealtime()+500)
            speech.cancelSemanticExcept(emptySet(),"semantic_rapid_motion")
        }
        try {
            assertTrue(start.await(5,TimeUnit.SECONDS)); instrumentation.waitForIdleSync()
            assertTrue(events.any{it.eventId=="semantic:obsolete" && it.phase==SpeechPlaybackPhase.CANCELLED})
            assertFalse(events.any{it.eventId=="semantic:obsolete" && it.phase==SpeechPlaybackPhase.ON_START})
        } finally { instrumentation.runOnMainSync{speech.close()} }
    }
}
