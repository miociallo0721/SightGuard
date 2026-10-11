package org.sightguard.app

import android.graphics.BitmapFactory
import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File

class GridFusionInstrumentedTest {
    @Test fun threeRealModelsFuseOnlyTheirSourceFrameAndAbsentImuIsUnknown() {
        val instrumentation=InstrumentationRegistry.getInstrumentation(); val context=instrumentation.targetContext
        val bitmap=instrumentation.context.assets.open("validation.jpg").use { BitmapFactory.decodeStream(it) }!!
        val key=SpatialFrameKey(1,0,4,100.0,bitmap.width,bitmap.height,SystemClock.elapsedRealtime())
        try {
            Detector(context).use { yolo -> SpatialDetector(context).use { semantics -> DepthDetector(context).use { depth ->
                val detected=yolo.detect(bitmap); val semantic=semantics.analyzeAll(bitmap,key); val predicted=depth.analyze(bitmap,key)
                assertTrue(detected.isNotEmpty()); assertTrue(semantic.cells.isNotEmpty()); assertEquals("VALID",predicted.quality)
                val grid=SpatialGridFusion().fuse(key,GridSemanticSource(key,semantic.cells,SpatialDetector.MODEL_SHA),GridDepthSource(key,64,64,FloatArray(4096) { i -> predicted.sample((i%64+.5f)/64,(i/64+.5f)/64) },depth.modelSha),GridDetectionSource(key,detected,yolo.modelSha),MotionEvidence(MotionQuality.UNKNOWN,"real photo has no synchronized IMU"))
                assertEquals(225,grid.cells.size); assertFalse(grid.productionDecisionChanged)
                assertTrue(grid.cells.all { it.sources.size==3 }); assertEquals(MotionQuality.UNKNOWN,grid.motion.quality)
                assertTrue(grid.cells.any { it.depth?.median!=null }); assertTrue(grid.cells.any { it.state=="CONFLICT" || it.state=="OBJECT" })
                val rows=JSONArray()
                grid.cells.forEach { cell -> rows.put(JSONObject().put("row",cell.row).put("col",cell.col).put("state",cell.state).put("quality",cell.quality).put("ground",cell.ground).put("wall",cell.wall).put("vegetation",cell.vegetation).put("bbox_occupancy_proxy",cell.occupancy).put("depth_median",cell.depth?.median).put("depth_relation",cell.depth?.relation).put("sources",JSONObject(cell.sources)).put("reasons",JSONArray(cell.reasons))) }
                File(context.filesDir,"android_grid_validation.json").writeText(JSONObject().put("source_frame_id","1:0:4").put("source_timestamp_ms",100.0).put("width",key.width).put("height",key.height).put("status",grid.state).put("motion",grid.motion.quality).put("production_decision_changed",false).put("cells",rows).toString(2))
            } } }
            ImuMonitor(context).use { imu ->
                Thread.sleep(500)
                assertEquals(MotionQuality.UNKNOWN,imu.evidence(key).quality)
                File(context.filesDir,"android_imu_inventory.txt").writeText(imu.diagnostic())
            }
        } finally { bitmap.recycle() }
    }
}
