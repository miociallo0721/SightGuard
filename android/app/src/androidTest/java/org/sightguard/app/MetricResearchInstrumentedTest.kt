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
import kotlin.math.abs
import kotlin.math.sqrt

class MetricResearchInstrumentedTest {
    @Test fun realMetricGraphsPreprocessingAndPublicRgbdRemainShadowOnly() {
        val instrumentation=InstrumentationRegistry.getInstrumentation()
        val context=instrumentation.targetContext
        val before=android.os.Debug.getPss()
        fun floats(name: String): FloatArray {
            val buffer=ByteBuffer.wrap(instrumentation.context.assets.open("metric_research/$name").use {it.readBytes()})
                .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            return FloatArray(buffer.remaining()).also(buffer::get)
        }
        val report=JSONObject().put("scope","Level_3_real_Android_ORT_public_RGBD_not_CameraX_or_tape_measurement")
            .put("device",android.os.Build.MODEL).put("api",android.os.Build.VERSION.SDK_INT)
            .put("pss_before_kb",before).put("automatic_metric_speech",false)
        val shapes=JSONArray();val scenes=JSONArray()
        MetricDepthDetector(context).use {detector->
            for(shape in MetricDepthMath.supportedShapes) {
                val suffix="${shape.height}x${shape.width}"
                val expectedInput=floats("input_$suffix.bin")
                val expected=floats("expected_$suffix.bin")
                val bitmap=instrumentation.context.assets.open("metric_research/source_$suffix.png").use {BitmapFactory.decodeStream(it)}!!
                val prepareStarted=SystemClock.elapsedRealtime()
                val prepared=detector.prepare(bitmap,shape)
                var maximumInput=0f
                prepared.forEachIndexed {i,value->maximumInput=maxOf(maximumInput,abs(value-expectedInput[i]))}
                val prepareMs=SystemClock.elapsedRealtime()-prepareStarted
                assertTrue("official cubic preprocessing delta=$maximumInput",maximumInput<.0003f)
                val started=SystemClock.elapsedRealtime();val actual=detector.infer(expectedInput,shape)
                val coldMs=SystemClock.elapsedRealtime()-started
                var maximum=0f
                actual.forEachIndexed {i,value->
                    maximum=maxOf(maximum,abs(value-expected[i]))
                    assertTrue("metric parity $suffix pixel=$i",value.isFinite() && abs(value-expected[i])<=.0003f+.001f*abs(expected[i]))
                }
                val warm=(0 until 3).map {val t=SystemClock.elapsedRealtime();detector.infer(expectedInput,shape);SystemClock.elapsedRealtime()-t}
                shapes.put(JSONObject().put("shape",suffix).put("cold_load_and_infer_ms",coldMs)
                    .put("warm_infer_ms",JSONArray(warm)).put("preprocess_ms",prepareMs)
                    .put("preprocess_max_abs_error",maximumInput).put("metric_max_abs_error_m",maximum)
                    .put("pss_kb",android.os.Debug.getPss()))
                bitmap.recycle()
            }
            val fixtures=JSONArray(instrumentation.context.assets.open("metric_research/scenes.json").bufferedReader().use {it.readText()})
            SpatialDetector(context).use {seg->
                for(index in 0 until fixtures.length()) {
                    val fixture=fixtures.getJSONObject(index)
                    val bitmap=instrumentation.context.assets.open("metric_research/${fixture.getString("image")}").use {BitmapFactory.decodeStream(it)}!!
                    val key=SpatialFrameKey(100,0,index,index*1000.0,bitmap.width,bitmap.height,SystemClock.elapsedRealtime())
                    val semantic=seg.analyzeAll(bitmap,key)
                    val semanticAge=SystemClock.elapsedRealtime()-key.capturedAtMs
                    val depth=detector.analyze(bitmap,key)
                    val record=WallMetricEstimator().estimate(key,semantic,depth,emptyList(),SystemClock.elapsedRealtime())
                    assertFalse(record.automaticAlertEnabled)
                    if(record.sourceAgeMs>500) {assertEquals("ABSTAIN",record.status);assertNull(record.predictedMetricDepth)}
                    val truth=floats(fixture.getString("truth"))
                    var absolute=0.0;var squared=0.0;var relative=0.0;var valid=0
                    for(y in 0 until bitmap.height) for(x in 0 until bitmap.width) {
                        val gt=truth[y*bitmap.width+x]
                        if(!gt.isFinite() || gt<=0) continue
                        val mx=x.toFloat()*(depth.shape.width-1)/(bitmap.width-1)
                        val my=y.toFloat()*(depth.shape.height-1)/(bitmap.height-1)
                        val x0=mx.toInt();val y0=my.toInt();val x1=(x0+1).coerceAtMost(depth.shape.width-1);val y1=(y0+1).coerceAtMost(depth.shape.height-1)
                        val a=depth.values[y0*depth.shape.width+x0]*(1-(mx-x0))+depth.values[y0*depth.shape.width+x1]*(mx-x0)
                        val b=depth.values[y1*depth.shape.width+x0]*(1-(mx-x0))+depth.values[y1*depth.shape.width+x1]*(mx-x0)
                        val prediction=a*(1-(my-y0))+b*(my-y0)
                        if(!prediction.isFinite() || prediction<=0) continue
                        val error=(prediction-gt).toDouble();absolute+=abs(error);squared+=error*error;relative+=abs(error)/gt;valid++
                    }
                    scenes.put(JSONObject().put("sample_id",fixture.getString("sample_id"))
                        .put("rgb_sha256",fixture.getString("rgb_sha256"))
                        .put("semantic_wall_cells",semantic.cells.count {it.classId==0})
                        .put("semantic_latency_ms",semantic.latencyMs).put("semantic_source_age_ms",semanticAge)
                        .put("metric_latency_ms",depth.latencyMs).put("wall",metricWallJson(record))
                        .put("valid_truth_pixels",valid).put("mae_m",if(valid>0) absolute/valid else JSONObject.NULL)
                        .put("rmse_m",if(valid>0) sqrt(squared/valid) else JSONObject.NULL)
                        .put("abs_rel",if(valid>0) relative/valid else JSONObject.NULL))
                    bitmap.recycle()
                }
            }
            report.put("weights_sha256",detector.weightsSha).put("shapes",shapes).put("public_rgbd_scenes",scenes)
                .put("pss_with_models_kb",android.os.Debug.getPss())
        }
        report.put("pss_after_close_kb",android.os.Debug.getPss())
        File(context.filesDir,"metric_research_android_validation.json").writeText(report.toString(2))
    }
}
