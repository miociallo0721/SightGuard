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

class SpatialModelInstrumentedTest {
    @Test fun realSegFormerSamePhotoTensorMatchesPytorchAndBitmapProducesRealGroundSemantics() {
        val instrument = InstrumentationRegistry.getInstrumentation(); val context = instrument.targetContext
        fun floats(asset: String): FloatArray {
            val bytes = instrument.context.assets.open(asset).use { it.readBytes() }
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            return FloatArray(buffer.remaining()).also(buffer::get)
        }
        val input = floats("segformer.tensor.bin"); val reference = floats("segformer.logits.bin")
        val loadStart = SystemClock.elapsedRealtime()
        val pssBeforeKb = android.os.Debug.getPss()
        SpatialDetector(context).use { detector ->
            val loaded = SystemClock.elapsedRealtime(); val actual = detector.infer(input); val completed = SystemClock.elapsedRealtime()
            var maximumDelta = 0.0; var samePixels = 0
            for (pixel in 0 until 4096) {
                var expectedClass = 0; var actualClass = 0
                for (cls in 0 until 150) {
                    val wanted = reference[cls * 4096 + pixel]; val got = actual[cls][pixel]
                    assertTrue("finite real logits", got.isFinite())
                    val delta = abs(got - wanted).toDouble(); maximumDelta = maxOf(maximumDelta, delta)
                    assertTrue("same input numeric class=$cls pixel=$pixel delta=$delta", delta <= .0003 + .001 * abs(wanted))
                    if (wanted > reference[expectedClass * 4096 + pixel]) expectedClass = cls
                    if (got > actual[actualClass][pixel]) actualClass = cls
                }
                if (expectedClass == actualClass) samePixels++
            }
            assertTrue("same-input argmax agreement", samePixels >= 4092)
            val bitmap = instrument.context.assets.open("validation.jpg").use { BitmapFactory.decodeStream(it) }!!
            try {
                val evidence = detector.analyze(bitmap, SpatialFrameKey(1, 0, 0, 0.0, bitmap.width, bitmap.height, SystemClock.elapsedRealtime()))
                assertTrue("real ground model output", evidence.cells.isNotEmpty())
                assertTrue(evidence.cells.all { it.classId in SpatialDetector.GROUND_CLASSES && it.confidence >= SpatialDetector.MIN_CONFIDENCE && it.left in 0f..1f && it.right in 0f..1f })
                val report = JSONObject().put("model_sha256", SpatialDetector.MODEL_SHA).put("load_ms", loaded - loadStart).put("same_input_infer_ms", completed - loaded)
                    .put("process_pss_before_kb", pssBeforeKb).put("process_pss_with_spatial_kb", android.os.Debug.getPss())
                    .put("max_absolute_error", maximumDelta).put("argmax_agreement", samePixels / 4096.0).put("bitmap_infer_ms", evidence.latencyMs).put("ground_cells", evidence.cells.size)
                    .put("bitmap_preprocessing", "Android bilinear, differs from PC INTER_AREA; same-tensor validation separately proved")
                File(context.filesDir, "android_segformer_validation.json").writeText(report.toString(2))
            } finally { bitmap.recycle() }
        }
    }
}
