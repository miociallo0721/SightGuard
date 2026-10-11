package org.sightguard.app

import kotlin.math.max

data class MetricDepthEvidence(val key: SpatialFrameKey, val values: FloatArray, val shape: MetricShape,
    val latencyMs: Long, val completedAtMs: Long, val modelSha: String,
    val quantity: String="predicted_optical_axis_z_m")

data class WallMetricRecord(val key: SpatialFrameKey, val status: String, val wallPresent: Boolean?,
    val predictedMetricDepth: Float?, val median: Float?, val p10: Float?, val p90: Float?,
    val validPixelFraction: Float, val wallCoverage: Float, val semanticConfidence: Float?,
    val wallPixels: Int, val componentCount: Int, val sourceAgeMs: Long, val modelLatencyMs: Long?,
    val rejectionReasons: List<String>, val modelSha: String?, val automaticAlertEnabled: Boolean=false,
    val motionQuality: String="UNKNOWN", val motionSync: String="MOTION_SYNC_UNKNOWN",
    val sourceAgeBasis: String="analyzer_receipt_or_decode_request_not_exposure_age",
    val generatedAtElapsedMs: Long=key.capturedAtMs+sourceAgeMs)

data class WallMetricConfig(val minimumConfidence: Float=.55f,val minimumComponentPixels: Int=64,
    val minimumRoiCoverage: Float=.08f,val minimumValidFraction: Float=.75f,
    val maximumSourceAgeMs: Long=500) {
    init {
        require(listOf(minimumConfidence,minimumRoiCoverage,minimumValidFraction).all {it.isFinite() && it>0 && it<=1})
        require(minimumComponentPixels>0 && maximumSourceAgeMs>0)
    }
}

/** Same-frame native-grid optical Z statistics. P10/P90 describe pixels, not accuracy bounds. */
class WallMetricEstimator(private val config: WallMetricConfig=WallMetricConfig()) {
    fun estimate(key: SpatialFrameKey, semantics: GroundEvidence?, depth: MetricDepthEvidence?,
                 detections: List<Detection>, nowMs: Long, sceneCurrent: Boolean=true,
                 currentMotion: MotionEvidence=MotionEvidence(MotionQuality.UNKNOWN,"motion unavailable")): WallMetricRecord {
        val reasons=mutableListOf<String>()
        val captureMs=if(key.timestampClock=="REALTIME" && (key.captureTimestampNs ?: 0)>0)
            key.captureTimestampNs!!/1_000_000 else null
        val age=nowMs-(captureMs ?: key.capturedAtMs)
        val ageBasis=if(captureMs!=null) "REALTIME_sensor_capture" else "analyzer_receipt_or_decode_request_not_exposure_age"
        if(key.timestampClock=="REALTIME" && captureMs==null) reasons.add("camera_capture_timestamp_missing")
        if(age !in 0..config.maximumSourceAgeMs) reasons.add("source_stale")
        if(!sceneCurrent) reasons.add("source_scene_changed")
        // Current rapid motion is a rejection signal, never an assertion of camera/IMU synchronization.
        if(currentMotion.quality==MotionQuality.RAPID) reasons.add("current_device_rapid_motion")
        if(semantics==null || semantics.key!=key) reasons.add("semantic_frame_missing_or_mismatch")
        if(semantics!=null && semantics.completedAtMs !in key.capturedAtMs..nowMs) reasons.add("semantic_completion_clock_invalid")
        val shape=depth?.shape ?: MetricShape(64,64)
        val count=shape.width*shape.height
        val mask=BooleanArray(count);val confidence=FloatArray(count)
        if(semantics?.key==key) semantics.cells.filter {it.classId==0 && it.confidence>=config.minimumConfidence && it.confidence<=1 && it.confidence.isFinite()}.forEach {cell->
            for(y in MetricDepthMath.pixelBounds(cell.top,cell.bottom,shape.height))
                for(x in MetricDepthMath.pixelBounds(cell.left,cell.right,shape.width)) {
                    val index=y*shape.width+x;mask[index]=true;confidence[index]=max(confidence[index],cell.confidence)
                }
        }
        detections.filter {it.confidence>=.35f}.forEach {d->
            for(y in MetricDepthMath.pixelBounds(d.top,d.bottom,shape.height))
                for(x in MetricDepthMath.pixelBounds(d.left,d.right,shape.width)) mask[y*shape.width+x]=false
        }
        val visited=BooleanArray(count);val queue=IntArray(count);var components=0
        for(start in mask.indices) if(mask[start] && !visited[start]) {
            var head=0;var tail=1;queue[0]=start;visited[start]=true
            while(head<tail) {
                val index=queue[head++];val x=index%shape.width;val y=index/shape.width
                for(dy in -1..1) for(dx in -1..1) {
                    val nx=x+dx;val ny=y+dy
                    if(nx !in 0 until shape.width || ny !in 0 until shape.height) continue
                    val next=ny*shape.width+nx
                    if(mask[next] && !visited[next]) {visited[next]=true;queue[tail++]=next}
                }
            }
            if(tail<config.minimumComponentPixels) for(i in 0 until tail) mask[queue[i]]=false else components++
        }
        val selected=mutableListOf<Int>();var roiPixels=0;var support=0.0
        for(y in 0 until shape.height) for(x in 0 until shape.width) {
            val nx=(x+.5f)/shape.width;val ny=(y+.5f)/shape.height
            if(nx !in .25f.. .75f || ny !in .2f.. .9f) continue
            roiPixels++;val index=y*shape.width+x
            if(mask[index]) {selected.add(index);support+=confidence[index]}
        }
        val coverage=selected.size.toFloat()/max(1,roiPixels)
        if(coverage<config.minimumRoiCoverage) reasons.add("insufficient_central_wall_support")
        val present=if(reasons.isEmpty()) true else if(reasons==listOf("insufficient_central_wall_support")) false else null
        val values=mutableListOf<Float>()
        if(depth!=null && (depth.completedAtMs !in key.capturedAtMs..nowMs || depth.latencyMs<0)) reasons.add("depth_completion_clock_invalid")
        if(depth!=null && depth.modelSha.isBlank()) reasons.add("depth_model_identity_missing")
        when {
            depth==null -> reasons.add("metric_depth_unavailable")
            depth.key!=key -> reasons.add("depth_frame_mismatch")
            depth.quantity!="predicted_optical_axis_z_m" -> reasons.add("depth_is_not_metric_z")
            depth.values.size!=count -> reasons.add("depth_geometry_mismatch")
            else -> selected.forEach {index->val value=depth.values[index];if(value.isFinite() && value>0 && value<19.8f) values.add(value)}
        }
        val fraction=values.size.toFloat()/max(1,selected.size)
        if(fraction<config.minimumValidFraction) reasons.add("insufficient_valid_metric_pixels")
        values.sort()
        fun quantile(q: Float): Float? {
            if(values.isEmpty()) return null
            val index=q*(values.size-1);val lower=index.toInt();val upper=(lower+1).coerceAtMost(values.lastIndex)
            return values[lower]+(index-lower)*(values[upper]-values[lower])
        }
        val median=quantile(.5f);val p10=quantile(.1f);val p90=quantile(.9f)
        if(median!=null && p10!=null && p90!=null && (p90-p10)/median>.8f) reasons.add("heterogeneous_wall_depth_or_multiple_planes")
        return WallMetricRecord(key,if(reasons.isEmpty()) "PREDICTED_METRIC_DEPTH" else "ABSTAIN",present,
            median.takeIf {reasons.isEmpty()},median,p10,p90,fraction,coverage,
            if(selected.isEmpty()) null else (support/selected.size).toFloat(),selected.size,components,
            age,depth?.latencyMs,reasons,depth?.modelSha,motionQuality=currentMotion.quality.name,
            sourceAgeBasis=ageBasis,generatedAtElapsedMs=nowMs)
    }
}
