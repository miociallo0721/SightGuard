package org.sightguard.app

import android.graphics.Bitmap
import android.os.SystemClock
import org.json.JSONObject
import kotlin.math.roundToInt

fun wallRecordJson(r: WallApproachRecord): JSONObject = JSONObject()
    .put("state",r.state.name).put("wall_present",r.wallPresent ?: JSONObject.NULL)
    .put("possible_near",r.possibleNear).put("approaching_candidate",r.approachingCandidate)
    .put("confidence",r.confidence).put("quality",if(r.abstainReason==null) "CANDIDATE" else "ABSTAIN")
    .put("abstain_reason",r.abstainReason).put("source_frame_key","${r.key.generation}:${r.key.epoch}:${r.key.frameIndex}")
    .put("source_ms",r.key.timestampMs).put("source_age_ms",r.sourceAgeMs).put("temporal_support",r.temporalSupport)
    .put("motion_quality",r.motion.quality.name).put("imu_synchronized",r.motion.synchronized)
    .put("imu_interval_start_ns",r.motion.intervalStartNs).put("imu_interval_end_ns",r.motion.intervalEndNs)
    .put("rotation_magnitude_rad",r.motion.rotationMagnitudeRad).put("depth_support",r.depthSupport)
    .put("flow_ms",r.flowLatencyMs).put("tracked_features",r.flow?.tracks?.size).put("inliers",r.flow?.inliers)
    .put("flow_budget_exceeded",r.flow?.budgetExceeded)
    .put("inlier_fraction",r.flow?.inlierFraction).put("coverage_quadrants",r.flow?.coverage)
    .put("visual_scale",r.flow?.scale).put("visual_rotation_rad",r.flow?.rotationRad)
    .put("center_translation_fraction",r.flow?.centerTranslation).put("fit_residual_px",r.flow?.residual)
    .put("ground_coverage",r.spatial?.groundCoverage).put("wall_grid_cells",r.spatial?.wallGridCells)
    .put("ground_grid_cells",r.spatial?.groundGridCells).put("mask_boundary_adjacencies",r.spatial?.boundaryAdjacencies)
    .put("wall_relative_depth_median",r.spatial?.wallDepthMedian).put("wall_relative_depth_q10",r.spatial?.wallDepthQ10).put("wall_relative_depth_q90",r.spatial?.wallDepthQ90)
    .put("automatic_alert_enabled",false).put("physical_phone_calibrated",false)

/** Worker-owned short image history, never a retained CameraX ImageProxy. */
class WallShadowAnalyzer(private val config: WallResearchConfig=WallResearchConfig()) {
    private val flow=WallFeatureFlow(config)
    private val rules=WallApproachRules(config)
    private val measure=SemanticStructureMeasure(EnvironmentConfig())
    private var image: GrayImage?=null
    private var semantics: GroundEvidence?=null
    val previousKey: SpatialFrameKey? get()=semantics?.key
    fun analyze(bitmap: Bitmap,evidence: GroundEvidence,depth: DepthEvidence?,motion: MotionEvidence,sceneCurrent: Boolean): WallApproachRecord {
        val started=SystemClock.elapsedRealtime()
        val scale=192f/maxOf(bitmap.width,bitmap.height)
        val small=Bitmap.createScaledBitmap(bitmap,(bitmap.width*scale).roundToInt().coerceAtLeast(16),(bitmap.height*scale).roundToInt().coerceAtLeast(16),true)
        val pixels=IntArray(small.width*small.height);small.getPixels(pixels,0,small.width,0,0,small.width,small.height)
        val gray=GrayImage(small.width,small.height,FloatArray(pixels.size) { i->val p=pixels[i];(.299f*((p shr 16) and 255)+.587f*((p shr 8) and 255)+.114f*(p and 255))/255 })
        if(small!==bitmap) small.recycle()
        val old=semantics
        val identity=old!=null && old.key.generation==evidence.key.generation && old.key.epoch==evidence.key.epoch && old.key.width==evidence.key.width && old.key.height==evidence.key.height && old.key.rotationDegrees==evidence.key.rotationDegrees
        val fresh=started-evidence.key.capturedAtMs in 0..config.maxSourceAgeMs
        val wall=measure.components(evidence.cells).any {it.kind==EnvironmentKind.WALL && it.supported}
        val motionUsable=motion.synchronized && motion.quality==MotionQuality.STABLE && motion.intervalStartNs!=null && (motion.gyroRadPerSecond ?: Float.MAX_VALUE)<=config.maxGyro && (motion.rotationMagnitudeRad ?: Float.MAX_VALUE)<=config.maxRotationRad
        val fit=if(motionUsable && identity && fresh && sceneCurrent && wall && image!=null && evidence.key.timestampMs-old!!.key.timestampMs in config.minPairMs..config.maxPairMs)
            flow.analyze(image!!,gray,old.cells) else null
        val now=SystemClock.elapsedRealtime()
        val depthSupport=when {
            depth==null -> "unavailable_or_not_scheduled"
            depth.key!=evidence.key -> "source_mismatch"
            now-depth.key.capturedAtMs !in 0..config.maxSourceAgeMs -> "stale"
            depth.quality!="VALID" -> "degenerate"
            else -> {
                val samples=evidence.cells.filter {it.classId==0 && it.confidence>=.55f}.map {depth.sample((it.left+it.right)/2,(it.top+it.bottom)/2)}.toFloatArray()
                val stats=RelativeDepthRules.summarize(samples,depth.sourceValues())
                "same_frame_relative_only:valid=${stats.validFraction};rank=${stats.framePercentile};not_metric"
            }
        }
        val grid=if(fresh && sceneCurrent) SpatialGridFusion().semanticSummary(evidence.key,evidence.cells) else null
        val ground=evidence.cells.filter {it.classId in SemanticEvidence.groundClassIds && it.confidence>=.55f}.sumOf {((it.right-it.left)*(it.bottom-it.top)).toDouble()}.toFloat()
        val depthStats=if(depth?.key==evidence.key && now-depth.key.capturedAtMs in 0..config.maxSourceAgeMs && depth.quality=="VALID")
            RelativeDepthRules.summarize(evidence.cells.filter {it.classId==0 && it.confidence>=.55f}.map {depth.sample((it.left+it.right)/2,(it.top+it.bottom)/2)}.toFloatArray(),depth.sourceValues()) else null
        val spatial=WallSpatialSupport(ground.takeIf {fresh && sceneCurrent},grid?.wallCells,grid?.groundCells,grid?.boundaryAdjacencies,depthStats?.median,depthStats?.q10,depthStats?.q90)
        val record=rules.evaluate(evidence.key,SystemClock.elapsedRealtime(),wall,fit,motion,depthSupport,SystemClock.elapsedRealtime()-started,sceneCurrent).copy(spatial=spatial)
        if(fresh && sceneCurrent && wall) { image=gray;semantics=evidence } else { image=null;semantics=null }
        return record
    }
}
