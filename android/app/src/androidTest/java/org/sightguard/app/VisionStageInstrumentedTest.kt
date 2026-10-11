package org.sightguard.app

import android.net.Uri
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import java.io.File

class VisionStageInstrumentedTest {
    @Test fun actualDecodedVideoRunsSameYoloAndRiskStageAndSeekResetsStream() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val file = File(context.cacheDir, "stage-video.mp4")
        instrumentation.context.assets.open("video-fixtures/bus_portrait.mp4").use { input -> file.outputStream().use(input::copyTo) }
        VideoFileSource(context, Uri.fromFile(file)).use { source ->
            VisionStage(context, "instrumented_video").use { stage ->
                for (time in listOf(0L, 300L, 600L)) {
                    val video = source.frameAt(time)
                    try {
                        val result = stage.process(video.bitmap, video.timestampMs, "fps_fallback")
                        assertTrue("Real ONNX output, no injected detections", result.raw.any { it.classId == 0 || it.classId == 5 })
                        assertEquals(time.toDouble(), result.records.timestampMs, 0.0)
                        assertEquals(video.bitmap.width, result.records.frameWidth)
                        assertTrue(result.risk.frame.observations.isNotEmpty())
                        if (time >= 300) assertTrue(result.risk.frame.events.any { it.confirmed })
                    } finally { video.bitmap.recycle() }
                }
                stage.reset()
                val seek = source.frameAt(100)
                try { val restarted = stage.process(seek.bitmap, 100, "fps_fallback"); assertEquals(0, restarted.records.frameIndex); assertTrue(restarted.risk.frame.events.none { it.confirmed }) }
                finally { seek.bitmap.recycle() }
            }
        }
    }
}
