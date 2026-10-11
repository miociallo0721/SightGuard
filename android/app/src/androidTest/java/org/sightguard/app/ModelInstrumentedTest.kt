package org.sightguard.app

import android.graphics.BitmapFactory
import android.os.Build
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs

/** Real packaged ONNX model and real bus photo. Never uses synthetic detector outputs. */
@RunWith(AndroidJUnit4::class)
class ModelInstrumentedTest {
    @Test fun realModelMatchesPythonReferenceOnBusPhoto() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val testContext = InstrumentationRegistry.getInstrumentation().context
        val image = testContext.assets.open("validation.jpg").use { BitmapFactory.decodeStream(it) }
        assertNotNull(image)
        val reference = JSONObject(testContext.assets.open("validation_reference.json").bufferedReader().use { it.readText() })
        val started = SystemClock.elapsedRealtime()
        val detections: List<Detection>
        var modelSha = ""
        Detector(context).use { detector ->
            modelSha = detector.modelSha
            detections = detector.detect(image)
        }
        val elapsed = SystemClock.elapsedRealtime() - started
        image.recycle()
        val output = JSONObject().put("schema_version", 1).put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
            .put("android_api", Build.VERSION.SDK_INT).put("runtime", "ONNX Runtime Android 1.24.3 CPU")
            .put("model_sha256", modelSha).put("cold_load_and_inference_ms", elapsed)
            .put("detections", JSONArray().apply { detections.forEach { d ->
                put(JSONObject().put("class_id", d.classId).put("confidence", d.confidence)
                    .put("bbox_normalized", JSONArray(listOf(d.left, d.top, d.right, d.bottom))))
            } })
        File(context.filesDir, "android_model_validation.json").writeText(output.toString(2))
        android.util.Log.i("SG_VALIDATION", output.toString())
        assertTrue("Actual ONNX inference must detect objects", detections.isNotEmpty())
        val expected = reference.getJSONObject("detections").getJSONArray("onnxruntime")
        val preprocessing = reference.getJSONObject("preprocessing")
        val width = preprocessing.getInt("resized_width").toFloat()
        val height = preprocessing.getInt("resized_height").toFloat()
        val padX = preprocessing.getInt("pad_left"); val padY = preprocessing.getInt("pad_top")
        val matches = JSONArray()
        for (index in 0 until expected.length()) {
            val row = expected.getJSONObject(index)
            if (row.getDouble("score") < Detector.MIN_SCORE) continue
            val box = row.getJSONArray("xyxy")
            val target = Detection(row.getInt("class_index"), row.getDouble("score").toFloat(),
                ((box.getDouble(0).toFloat() - padX) / width).coerceIn(0f, 1f),
                ((box.getDouble(1).toFloat() - padY) / height).coerceIn(0f, 1f),
                ((box.getDouble(2).toFloat() - padX) / width).coerceIn(0f, 1f),
                ((box.getDouble(3).toFloat() - padY) / height).coerceIn(0f, 1f))
            val best = detections.filter { it.classId == target.classId }.maxByOrNull { iou(it, target) }
            assertNotNull("Missing Python class ${target.classId}", best)
            val overlap = iou(best!!, target)
            // Android and OpenCV bilinear kernels differ slightly; compare objects/geometry, not bits.
            assertTrue("Insufficient bbox agreement: $overlap", overlap >= 0.80f)
            assertTrue("Confidence mismatch", abs(best.confidence - target.confidence) <= 0.15f)
            matches.put(JSONObject().put("class_id", best.classId).put("iou", overlap).put("confidence_delta", abs(best.confidence - target.confidence)))
        }
        assertEquals("Object count differs from Python reference", matches.length(), detections.size)
        output.put("comparison", matches)
        File(context.filesDir, "android_model_validation.json").writeText(output.toString(2))
        android.util.Log.i("SG_VALIDATION", output.toString())
    }

    private fun iou(a: Detection, b: Detection): Float {
        val intersection = maxOf(0f, minOf(a.right, b.right) - maxOf(a.left, b.left)) * maxOf(0f, minOf(a.bottom, b.bottom) - maxOf(a.top, b.top))
        val union = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - intersection
        return if (union > 0f) intersection / union else 0f
    }
}
