package org.sightguard.app

import android.graphics.BitmapFactory
import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class YoloeModelInstrumentedTest {
    @Test fun actualExportParityFurnitureAndSerialPerformance() {
        val ins=InstrumentationRegistry.getInstrumentation();val context=ins.targetContext
        fun floats(name: String): FloatArray {
            val bytes=ins.context.assets.open(name).use {it.readBytes()}
            val b=ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            return FloatArray(b.remaining()).also {b.get(it)}
        }
        val input=floats("yoloe_input.bin");val reference=floats("yoloe_output.bin")
        val bus=ins.context.assets.open("validation.jpg").use {BitmapFactory.decodeStream(it)}!!
        val table=ins.context.assets.open("yoloe_table.jpg").use {BitmapFactory.decodeStream(it)}!!
        val report=JSONObject().put("device",android.os.Build.MODEL).put("api",android.os.Build.VERSION.SDK_INT)
        val runs=JSONArray()
        try {
            for(useNew in listOf(false,true)) {
                val before=android.os.Debug.getPss();val started=SystemClock.elapsedRealtime()
                Detector(context,experimental=useNew).use {detector ->
                    val load=SystemClock.elapsedRealtime()-started
                    if(useNew) {
                        val actual=detector.inferTensor(input);assertEquals(reference.size,actual.sumOf {it.size})
                        var maxDelta=0f;var matched=0
                        reference.asList().chunked(6).filter {it[4]>=.35f}.forEach {row ->
                            val best=actual.filter {it[5]==row[5]}.minByOrNull {r->(0..3).maxOf {k->kotlin.math.abs(r[k]-row[k])}}
                            assertNotNull(best)
                            val delta=(0..3).maxOf {k->kotlin.math.abs(best!![k]-row[k])}
                            assertTrue("same-tensor box parity $delta",delta<.03f)
                            assertEquals(row[4],best!![4],.001f);maxDelta=maxOf(maxDelta,delta);matched++
                        }
                        assertEquals(matched,actual.count {it[4]>=.35f});report.put("parity_max_bbox_delta",maxDelta).put("parity_matches",matched)
                        val b=detector.detect(bus);assertTrue(b.any {it.classId==0});assertTrue(b.any {it.classId==5})
                        val t=detector.detect(table);assertTrue("actual furniture table missing",t.any {it.classId==60 && it.sourceClassId in setOf(80,81)})
                        report.put("furniture",JSONArray(t.map {JSONObject().put("class_id",it.classId).put("source_class_id",it.sourceClassId).put("score",it.confidence)}))
                    }
                    detector.detect(bus)
                    val times=mutableListOf<Long>()
                    repeat(12) { val begin=SystemClock.elapsedRealtime();detector.detect(if(it%2==0) bus else table);times.add(SystemClock.elapsedRealtime()-begin) }
                    runs.put(JSONObject().put("experimental",useNew).put("sha256",detector.modelSha).put("load_ms",load)
                        .put("times_ms",JSONArray(times)).put("pss_before_kb",before).put("pss_loaded_kb",android.os.Debug.getPss()))
                }
            }
        } finally {
            bus.recycle();table.recycle();report.put("runs",runs)
            File(context.filesDir,"yoloe-android-validation.json").writeText(report.toString(2))
            android.util.Log.i("SG_YOLOE_VALIDATION",report.toString())
        }
    }
}
