package org.sightguard.app

import android.graphics.BitmapFactory
import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.abs

/** Same-tensor PyTorch parity plus real bitmap latency; no camera FPS claim. */
class SegFormerResolutionInstrumentedTest {
    @Test fun actualResolutionContractMatchesPytorchAndPreservesCoordinates() {
        val context=InstrumentationRegistry.getInstrumentation().targetContext
        val size=InstrumentationRegistry.getArguments().getString("semantic_size","384").toInt()
        val contract=SegFormerModels.contract(size);val pixels=contract.outputSize*contract.outputSize
        val root=File(context.filesDir,"furniture")
        val report=JSONObject(File(root,"export-validation.json").readText())
        assertEquals(contract.sha256,report.getJSONObject("onnx").getString("sha256"))
        fun floats(name: String, referenceKey: String): FloatArray {
            val bytes=File(root,name).readBytes()
            val metadata=report.getJSONObject("referenceBinaries").getJSONObject(referenceKey)
            assertEquals(metadata.getInt("sizeBytes"),bytes.size)
            assertEquals(metadata.getString("sha256"),MessageDigest.getInstance("SHA-256").digest(bytes).joinToString(""){"%02x".format(it)})
            val values=FloatArray(bytes.size/4)
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(values)
            return values
        }
        val input=floats("input.tensor.bin","inputTensor");val reference=floats("reference.logits.bin","pytorchLogits")
        assertEquals(3*size*size,input.size);assertEquals(150*pixels,reference.size)
        val manifest=JSONObject(File(root,"manifest.json").readText()).getJSONArray("cases")
        val case=(0 until manifest.length()).map {manifest.getJSONObject(it)}.first {it.getJSONObject("gt_fractions").getDouble("CHAIR")>=.015}
        val imageBytes=File(root,case.getString("image")).readBytes()
        assertEquals(case.getString("image_sha256"),MessageDigest.getInstance("SHA-256").digest(imageBytes).joinToString(""){"%02x".format(it)})
        val bitmap=BitmapFactory.decodeByteArray(imageBytes,0,imageBytes.size)
        val latencies=JSONArray();val before=android.os.Debug.getPss();val started=SystemClock.elapsedRealtime()
        var loadMs=0L;var peakPss=before;var maxError=0f
        try {SpatialDetector(context,2,size).use {detector->
            loadMs=SystemClock.elapsedRealtime()-started;peakPss=maxOf(peakPss,android.os.Debug.getPss())
            val logits=detector.infer(input);assertEquals(150,logits.size)
            var agree=0
            for(pixel in 0 until pixels) {
                var actualBest=0;var expectedBest=0
                for(cls in 0 until 150) {
                    val expected=reference[cls*pixels+pixel];val actual=logits[cls][pixel]
                    assertTrue(actual.isFinite());maxError=maxOf(maxError,abs(actual-expected))
                    assertEquals(expected,actual,.0001f+.001f*abs(expected))
                    if(actual>logits[actualBest][pixel]) actualBest=cls
                    if(expected>reference[expectedBest*pixels+pixel]) expectedBest=cls
                }
                if(actualBest==expectedBest) agree++
            }
            assertTrue("argmax agreement",agree>=pixels-10)
            for(index in 0..2) {
                val captured=SystemClock.elapsedRealtime()
                val evidence=detector.analyzeAll(bitmap,SpatialFrameKey(1,0,index,captured.toDouble(),bitmap.width,bitmap.height,captured))
                assertEquals(contract.sha256,evidence.modelSha);assertEquals(size,evidence.inputSize)
                assertTrue(evidence.cells.isNotEmpty())
                assertTrue(evidence.cells.all {it.left>=0 && it.top>=0 && it.right<=1 && it.bottom<=1 && it.right>it.left && it.bottom>it.top})
                assertTrue(evidence.cells.sumOf {((it.right-it.left)*(it.bottom-it.top)).toDouble()}<=1.00001)
                latencies.put(evidence.latencyMs);peakPss=maxOf(peakPss,android.os.Debug.getPss())
            }
        }} finally {
            bitmap.recycle()
            File(context.filesDir,"segformer${size}-runtime.json").writeText(JSONObject().put("device",android.os.Build.MODEL)
                .put("model_sha256",contract.sha256).put("load_ms",loadMs).put("bitmap_latencies_ms",latencies)
                .put("max_absolute_logit_error",maxError).put("before_pss_kb",before).put("peak_sampled_pss_kb",peakPss)
                .put("scope","real Android CPU model and bitmap; no concurrent YOLO or live camera acceptance").toString(2))
        }
    }
}
