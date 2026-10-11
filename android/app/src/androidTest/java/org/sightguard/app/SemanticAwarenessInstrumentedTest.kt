package org.sightguard.app

import android.graphics.*
import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

class SemanticAwarenessInstrumentedTest {
    @Test fun actualRoomDoorTreesAndSidewalkAndSameTensorAgreement() {
        val instrumentation=InstrumentationRegistry.getInstrumentation(); val context=instrumentation.targetContext
        val report=JSONObject(); val photos=JSONArray()
        fun floats(name:String):FloatArray {
            val buffer=ByteBuffer.wrap(instrumentation.context.assets.open(name).use { it.readBytes() }).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            return FloatArray(buffer.remaining()).also(buffer::get)
        }
        SpatialDetector(context).use { detector ->
            val input=floats("v03-semantic.tensor.bin"); val reference=floats("v03-semantic.logits.bin"); val output=detector.infer(input)
            var maximum=0f; var agreement=0
            for (pixel in 0 until 4096) {
                var actualClass=0; var expectedClass=0
                for (cls in 0 until 150) {
                    val wanted=reference[cls*4096+pixel]; val got=output[cls][pixel]
                    assertTrue(got.isFinite()); maximum=maxOf(maximum,abs(got-wanted))
                    assertTrue(abs(got-wanted)< .0003f + .001f*abs(wanted))
                    if (got>output[actualClass][pixel]) actualClass=cls
                    if (wanted>reference[expectedClass*4096+pixel]) expectedClass=cls
                }
                if(actualClass==expectedClass) agreement++
            }
            assertTrue(agreement>=4092)
            report.put("same_tensor_max_absolute_error",maximum).put("argmax_agreement",agreement/4096.0)
            for ((index,sample) in listOf("room","door","outdoor","sidewalk").withIndex()) {
                val bitmap=instrumentation.context.assets.open("v03-$sample.jpg").use { BitmapFactory.decodeStream(it) }!!
                try {
                    val key=SpatialFrameKey(1,0,index,0.0,bitmap.width,bitmap.height,SystemClock.elapsedRealtime())
                    val evidence=detector.analyzeAll(bitmap,key)
                    val counts=evidence.cells.groupingBy { it.classId }.eachCount()
                    when(sample) {
                        "room" -> { assertTrue((counts[0]?:0)>10); assertTrue((counts[19]?:0)>10); assertTrue((counts[15]?:0)>0) }
                        "door" -> assertTrue((counts[14]?:0)>10)
                        "outdoor" -> assertTrue((counts[4]?:0)>10)
                        "sidewalk" -> assertTrue((counts[11]?:0)>10)
                    }
                    val summary=SemanticEvidence.summarize(evidence.cells,emptyList(),SemanticSourceQuality())
                    assertTrue(summary.regions.isNotEmpty()); assertNotEquals("当前环境信息不足。",summary.overview)
                    val overlay=bitmap.copy(Bitmap.Config.ARGB_8888,true); val canvas=Canvas(overlay)
                    val paint=Paint().apply { color=Color.argb(95,255,150,70) }
                    evidence.cells.forEach { cell -> canvas.drawRect(cell.left*bitmap.width,cell.top*bitmap.height,cell.right*bitmap.width,cell.bottom*bitmap.height,paint) }
                    File(context.filesDir,"semantic_$sample.png").outputStream().use { overlay.compress(Bitmap.CompressFormat.PNG,100,it) }; overlay.recycle()
                    photos.put(JSONObject().put("sample",sample).put("frame_index",index).put("width",bitmap.width).put("height",bitmap.height)
                        .put("inference_ms",evidence.latencyMs).put("accepted_class_counts",JSONObject(counts.mapKeys { it.key.toString() })).put("overview",summary.overview))
                } finally { bitmap.recycle() }
            }
        }
        report.put("model_sha256",SpatialDetector.MODEL_SHA).put("actual_samples",photos).put("source_provenance","v03-semantics-python/sources.json; public photographs, no ground truth labels used")
        File(context.filesDir,"android_semantic_awareness_validation.json").writeText(report.toString(2))
    }
}
