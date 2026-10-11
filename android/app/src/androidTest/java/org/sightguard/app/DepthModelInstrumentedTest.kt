package org.sightguard.app

import android.graphics.BitmapFactory
import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

class DepthModelInstrumentedTest {
    @Test fun realDepthMatchesPythonTensorAndRealBitmapProvidesRelativeRegions() {
        val instrumentation = InstrumentationRegistry.getInstrumentation(); val context = instrumentation.targetContext
        fun floats(name: String): FloatArray {
            val buffer = ByteBuffer.wrap(instrumentation.context.assets.open(name).use { it.readBytes() }).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            return FloatArray(buffer.remaining()).also(buffer::get)
        }
        val input = floats("depth_input.tensor.bin"); val reference = floats("depth_output.tensor.bin")
        val before = android.os.Debug.getPss(); val start = SystemClock.elapsedRealtime()
        val report = JSONObject()
        DepthDetector(context).use { detector ->
            val loaded = SystemClock.elapsedRealtime(); val actual = detector.infer(input); val inferred = SystemClock.elapsedRealtime()
            assertEquals(reference.size,actual.size)
            var maximum = 0f
            actual.forEachIndexed { index,value ->
                assertTrue(value.isFinite()); maximum = maxOf(maximum,abs(value-reference[index]))
                assertTrue("real depth index=$index",abs(value-reference[index]) <= .0003f + .001f * abs(reference[index]))
            }
            val warmTimes = (0 until 4).map { val t=SystemClock.elapsedRealtime(); detector.infer(input); SystemClock.elapsedRealtime()-t }
            val image = instrumentation.context.assets.open("validation.jpg").use { BitmapFactory.decodeStream(it) }!!
            try {
                val key = SpatialFrameKey(1,0,0,0.0,image.width,image.height,SystemClock.elapsedRealtime())
                val depth = detector.analyze(image,key)
                assertEquals("VALID",depth.quality); assertEquals(280*280,depth.values.size)
                Detector(context).use { yolo ->
                    val detections = yolo.detect(image)
                    assertTrue(detections.isNotEmpty())
                    val regions = org.json.JSONArray()
                    for (d in detections) {
                        val stats=depth.region(d); assertNotNull(stats.median); assertTrue(stats.validFraction>=.8f)
                        regions.put(JSONObject().put("class_id",d.classId).put("median",stats.median).put("q10",stats.q10).put("q90",stats.q90).put("valid_fraction",stats.validFraction).put("frame_percentile",stats.framePercentile).put("relation",stats.relation))
                    }
                    report.put("bbox_inset_regions",regions)
                }
                val heat=depth.heatmap(); File(context.filesDir,"depth_bus_heatmap.png").outputStream().use { heat.compress(android.graphics.Bitmap.CompressFormat.PNG,100,it) }; heat.recycle()
                report.put("model_sha256",detector.modelSha).put("load_ms",loaded-start).put("single_infer_ms",inferred-loaded).put("warm_infer_ms",org.json.JSONArray(warmTimes))
                    .put("bitmap_infer_ms",depth.latencyMs).put("max_absolute_error",maximum).put("pss_before_kb",before).put("pss_with_models_kb",android.os.Debug.getPss()).put("quality",depth.quality)
                    .put("depth_semantics","relative_inverse_depth_larger_is_nearer").put("bitmap_preprocessing","Android bilinear differs from Python INTER_AREA; same tensor checked separately")
            } finally { image.recycle() }
        }
        report.put("pss_after_session_close_kb",android.os.Debug.getPss())
        File(context.filesDir,"android_depth_validation.json").writeText(report.toString(2))
    }
}
