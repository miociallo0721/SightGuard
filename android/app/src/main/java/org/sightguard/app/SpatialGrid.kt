package org.sightguard.app

import kotlin.math.max
import kotlin.math.min

enum class MotionQuality { STABLE, MODERATE, RAPID, LIMITED, UNKNOWN }
data class MotionEvidence(val quality: MotionQuality, val reason: String, val gyroRadPerSecond: Float? = null, val pitchDegrees: Float? = null, val rollDegrees: Float? = null, val synchronized: Boolean = false, val sampleCount: Int = 0)
data class MotionConfig(val stableGyro: Float = .15f, val rapidGyro: Float = 1.5f, val maxSampleGapNs: Long = 200_000_000L, val moderateOrientationDelta: Float = 10f, val rapidOrientationDelta: Float = 30f)
object MotionRules {
    fun classify(gyro: Float?, gapNs: Long?, synchronized: Boolean, pitch: Float? = null, roll: Float? = null, config: MotionConfig = MotionConfig(), orientationDeltaDegrees: Float? = null): MotionEvidence {
        if (!synchronized) return MotionEvidence(MotionQuality.UNKNOWN,"camera/IMU clock unproven",gyro,pitch,roll)
        if (gyro == null || !gyro.isFinite() || gapNs == null || gapNs < 0 || gapNs > config.maxSampleGapNs) return MotionEvidence(MotionQuality.LIMITED,"gyro missing, invalid or stale",gyro,pitch,roll,true)
        val quality = when {
            gyro >= config.rapidGyro || orientationDeltaDegrees?.let { it >= config.rapidOrientationDelta } == true -> MotionQuality.RAPID
            gyro > config.stableGyro || orientationDeltaDegrees?.let { it >= config.moderateOrientationDelta } == true -> MotionQuality.MODERATE
            else -> MotionQuality.STABLE
        }
        return MotionEvidence(quality,"measured gyro; no metric pose",gyro,pitch,roll,true)
    }
}

data class GridConfig(val rows: Int = 15, val cols: Int = 15, val minConfidence: Float = .35f, val supportThreshold: Float = .25f, val occupancyThreshold: Float = .25f) {
    init { require(rows in 1..128 && cols in 1..128); require(listOf(minConfidence,supportThreshold,occupancyThreshold).all { it.isFinite() && it in 0f..1f }) }
}
data class GridSemanticSource(val key: SpatialFrameKey, val cells: List<SemanticCell>, val modelSha: String)
data class GridDepthSource(val key: SpatialFrameKey, val width: Int, val height: Int, val values: FloatArray, val modelSha: String)
data class GridDetectionSource(val key: SpatialFrameKey, val detections: List<Detection>, val modelSha: String)
data class GridCell(val row: Int, val col: Int, val left: Float, val top: Float, val right: Float, val bottom: Float, val ground: Float?, val wall: Float?, val vegetation: Float?, val occupancy: Float?, val depth: DepthStatistics?, val state: String, val quality: String, val reasons: List<String>, val sources: Map<String,String>)
data class SpatialGridEvidence(val key: SpatialFrameKey, val cells: List<GridCell>, val motion: MotionEvidence, val state: String, val failures: List<String>, val productionDecisionChanged: Boolean = false)

/** Stateless 15x15 image evidence. PC thresholds retained; no temporal reuse or risk imports. */
class SpatialGridFusion(private val config: GridConfig = GridConfig()) {
    fun fuse(key: SpatialFrameKey, semantics: GridSemanticSource?, depth: GridDepthSource?, detections: GridDetectionSource?, motion: MotionEvidence): SpatialGridEvidence {
        require(key.width >= config.cols && key.height >= config.rows) { "image too small for grid" }
        val failures = mutableListOf<String>()
        fun same(candidate: SpatialFrameKey?, name: String): Boolean {
            val valid = candidate == key
            if (!valid) failures += "$name:${if(candidate==null) "missing" else "source_identity_time_or_size_mismatch"}"
            return valid
        }
        val semanticValid = same(semantics?.key,"semantic")
        var depthValid = same(depth?.key,"depth")
        val detectionValid = same(detections?.key,"detection")
        if (depthValid && (depth!!.width < 1 || depth.height < 1 || depth.values.size != depth.width * depth.height)) { failures += "depth:invalid_dimensions"; depthValid = false }
        val finiteScene = if(depthValid) depth!!.values.filter { it.isFinite() && it >= 0 }.sorted() else emptyList()
        val cells = mutableListOf<GridCell>()
        for (row in 0 until config.rows) for (col in 0 until config.cols) {
            // Pixel half-open partition, matching PC grid_bounds for odd source dimensions.
            val l=(col*key.width/config.cols).toFloat()/key.width; val r=((col+1)*key.width/config.cols).toFloat()/key.width
            val t=(row*key.height/config.rows).toFloat()/key.height; val b=((row+1)*key.height/config.rows).toFloat()/key.height
            val area=(r-l)*(b-t); val reasons=mutableListOf<String>(); val sources=linkedMapOf<String,String>()
            fun overlap(left: Float,top: Float,right: Float,bottom: Float)=max(0f,min(r,right)-max(l,left))*max(0f,min(b,bottom)-max(t,top)) / area
            var ground=if(semanticValid) 0f else null; var wall=if(semanticValid) 0f else null; var vegetation=if(semanticValid) 0f else null
            if(semanticValid) {
                sources["semantic"]=semantics!!.modelSha
                semantics.cells.forEach { cell ->
                    if(cell.confidence.isFinite() && cell.confidence>=config.minConfidence && listOf(cell.left,cell.top,cell.right,cell.bottom).all { it.isFinite() } && cell.right>cell.left && cell.bottom>cell.top) {
                        val fraction=overlap(cell.left,cell.top,cell.right,cell.bottom)
                        if(cell.classId in SemanticEvidence.groundClassIds) ground=ground!!+fraction
                        if(cell.classId==0) wall=wall!!+fraction
                        if(cell.classId in setOf(4,9,17)) vegetation=vegetation!!+fraction
                    }
                }
                ground=ground!!.coerceAtMost(1f); wall=wall!!.coerceAtMost(1f); vegetation=vegetation!!.coerceAtMost(1f)
            } else reasons += "semantic_unknown"
            var occupancy:Float?=null
            if(detectionValid) {
                sources["detection"]=detections!!.modelSha
                occupancy=detections.detections.filter { it.confidence.isFinite() && it.confidence>=config.minConfidence && listOf(it.left,it.top,it.right,it.bottom).all { v -> v.isFinite() } && it.right>it.left && it.bottom>it.top }
                    .maxOfOrNull { overlap(it.left,it.top,it.right,it.bottom) }?.coerceAtMost(1f) ?: 0f
            } else reasons += "detection_unknown"
            var stats:DepthStatistics?=null
            if(depthValid && finiteScene.isNotEmpty()) {
                sources["depth"]=depth!!.modelSha
                val samples=FloatArray(36) { i ->
                    val x=((l+(r-l)*(i%6+.5f)/6)*depth.width).toInt().coerceIn(0,depth.width-1)
                    val y=((t+(b-t)*(i/6+.5f)/6)*depth.height).toInt().coerceIn(0,depth.height-1)
                    depth.values[y*depth.width+x]
                }
                stats=summarize(samples,finiteScene)
                if(stats.framePercentile==null || stats.validFraction<.8f) reasons += "depth_unknown"
            } else reasons += "depth_unknown"
            if(motion.quality in setOf(MotionQuality.RAPID,MotionQuality.LIMITED,MotionQuality.UNKNOWN)) reasons += "motion_${motion.quality}"
            val groundSupported=(ground?:0f)>=config.supportThreshold; val occupied=(occupancy?:0f)>=config.occupancyThreshold
            val conflict=groundSupported && occupied
            if(conflict) reasons += "ground_and_bbox_overlap"
            val state=when { conflict->"CONFLICT"; occupied->"OBJECT"; (wall?:0f)>=config.supportThreshold->"WALL"; groundSupported->"GROUND"; (vegetation?:0f)>=config.supportThreshold->"VEGETATION"; else->"UNKNOWN" }
            if(state=="UNKNOWN") reasons += "insufficient_cell_support"
            val quality=when { motion.quality==MotionQuality.RAPID->"DEGRADED"; state=="UNKNOWN"->"UNKNOWN"; reasons.isNotEmpty()->"LIMITED"; else->"VALID" }
            cells += GridCell(row,col,l,t,r,b,ground,wall,vegetation,occupancy,stats,state,quality,reasons,sources)
        }
        return SpatialGridEvidence(key,cells,motion,if(failures.isEmpty() && motion.synchronized) "SAME_FRAME_RESEARCH" else "PARTIAL",failures)
    }
    private fun summarize(values: FloatArray, scene: List<Float>): DepthStatistics {
        val valid=values.filter { it.isFinite() && it>=0 }.sorted(); val fraction=valid.size.toFloat()/values.size
        if(valid.isEmpty()) return DepthStatistics(fraction,null,null,null,null,"UNKNOWN")
        val median=RelativeDepthRules.quantile(valid,.5f)
        val spread=RelativeDepthRules.quantile(scene,.99f)-RelativeDepthRules.quantile(scene,.01f)
        var lo=0; var hi=scene.size
        while(lo<hi) { val mid=(lo+hi)/2; if(scene[mid]<median) lo=mid+1 else hi=mid }
        val rank=if(spread>RelativeDepthRules.MIN_RELATIVE_SPREAD*maxOf(scene.last(),1e-6f)) lo.toFloat()/scene.size else null
        val relation=when { fraction<.8f || rank==null->"深度不确定"; rank>=.7f->"相对较近"; rank<=.3f->"相对较远"; else->"场景中间范围" }
        return DepthStatistics(fraction,median,RelativeDepthRules.quantile(valid,.1f),RelativeDepthRules.quantile(valid,.9f),rank,relation)
    }
}
