package org.sightguard.app

import kotlin.math.*

enum class WallEvidenceState { WALL_PRESENT, WALL_POSSIBLY_NEAR, WALL_APPROACHING, UNKNOWN, ABSTAIN }
data class WallResearchConfig(
    val maxSourceAgeMs: Long=500, val minPairMs: Double=100.0, val maxPairMs: Double=1600.0,
    val maxFeatures: Int=48, val minFeatures: Int=16, val featureScore: Float=.001f,
    val maxPatchResidual: Float=.08f, val maxFitResidual: Float=1.25f,
    val minInlierFraction: Float=.7f, val maxGyro: Float=.08f, val maxRotationRad: Float=.015f,
    val minExpansion: Float=1.012f, val maxExpansion: Float=1.15f, val confirmationWindows: Int=3,val maxFlowMs: Long=40,
) {
    init { require(maxSourceAgeMs>0 && minPairMs>0 && maxPairMs>minPairMs && minFeatures>=8 && maxFeatures>=minFeatures && confirmationWindows>=3 && maxFlowMs in 1..100) }
}
data class GrayImage(val width: Int,val height: Int,val values: FloatArray) {
    init { require(width>8 && height>8 && values.size==width*height) }
    fun at(x: Float,y: Float): Float {
        val ix=x.toInt().coerceIn(0,width-2); val iy=y.toInt().coerceIn(0,height-2)
        val fx=(x-ix).coerceIn(0f,1f); val fy=(y-iy).coerceIn(0f,1f)
        return values[iy*width+ix]*(1-fx)*(1-fy)+values[iy*width+ix+1]*fx*(1-fy)+values[(iy+1)*width+ix]*(1-fx)*fy+values[(iy+1)*width+ix+1]*fx*fy
    }
    fun half(): GrayImage {
        val w=width/2; val h=height/2
        return GrayImage(w,h,FloatArray(w*h) { i-> val x=i%w*2; val y=i/w*2
            (values[y*width+x]+values[y*width+x+1]+values[(y+1)*width+x]+values[(y+1)*width+x+1])/4 })
    }
}
data class FlowPoint(val x: Float,val y: Float,val nx: Float,val ny: Float,val residual: Float)
data class FlowFit(val tracks: List<FlowPoint>,val inliers: Int,val inlierFraction: Float,val coverage: Int,
    val scale: Float?,val rotationRad: Float?,val centerTranslation: Float?,val residual: Float?,val budgetExceeded: Boolean=false)

/** Sparse Shi–Tomasi features, two-level iterative LK, forward/backward check and similarity RANSAC.
 * A similarity fit cannot prove physical translation or a single plane; callers must abstain accordingly.
 */
class WallFeatureFlow(private val config: WallResearchConfig=WallResearchConfig()) {
    fun analyze(previous: GrayImage,current: GrayImage,mask: List<SemanticCell>): FlowFit {
        val deadline=System.nanoTime()+config.maxFlowMs*1_000_000
        fun budgetFailure()=FlowFit(emptyList(),0,0f,0,null,null,null,null,true)
        if(previous.width!=current.width || previous.height!=current.height) return FlowFit(emptyList(),0,0f,0,null,null,null,null)
        val wallPixels=BooleanArray(previous.width*previous.height)
        mask.filter {it.classId==0 && it.confidence>=.55f}.forEach { c->
            for(y in (c.top*previous.height).toInt().coerceAtLeast(0) until ceil(c.bottom*previous.height).toInt().coerceAtMost(previous.height))
                for(x in (c.left*previous.width).toInt().coerceAtLeast(0) until ceil(c.right*previous.width).toInt().coerceAtMost(previous.width)) wallPixels[y*previous.width+x]=true
        }
        fun wall(x: Int,y: Int): Boolean = wallPixels[y*previous.width+x]
        val candidates=mutableListOf<Triple<Float,Float,Float>>()
        for(y in 7 until previous.height-7 step 3) for(x in 7 until previous.width-7 step 3) {
            if(System.nanoTime()>deadline) return budgetFailure()
            if(!wall(x,y) || !wall(x-4,y-4) || !wall(x+4,y+4)) continue
            var xx=0f;var yy=0f;var xy=0f
            for(dy in -1..1) for(dx in -1..1) {
                val gx=(previous.at((x+dx+1).toFloat(),(y+dy).toFloat())-previous.at((x+dx-1).toFloat(),(y+dy).toFloat()))/2
                val gy=(previous.at((x+dx).toFloat(),(y+dy+1).toFloat())-previous.at((x+dx).toFloat(),(y+dy-1).toFloat()))/2
                xx+=gx*gx;yy+=gy*gy;xy+=gx*gy
            }
            val score=(xx+yy-sqrt((xx-yy)*(xx-yy)+4*xy*xy))/2
            if(score>=config.featureScore) candidates.add(Triple(x.toFloat(),y.toFloat(),score))
        }
        val features=mutableListOf<Triple<Float,Float,Float>>()
        for(point in candidates.sortedByDescending {it.third}) if(features.none { hypot(it.first-point.first,it.second-point.second)<9f }) {
            features.add(point); if(features.size>=config.maxFeatures) break
        }
        val pHalf=previous.half();val cHalf=current.half()
        fun track(a: GrayImage,b: GrayImage,ah: GrayImage,bh: GrayImage,x: Float,y: Float): Triple<Float,Float,Float>? {
            val coarse=lk(ah,bh,x/2,y/2,0f,0f) ?: return null
            return lk(a,b,x,y,coarse.first*2,coarse.second*2)
        }
        val points=features.mapNotNull { (x,y,_) ->
            if(System.nanoTime()>deadline) return budgetFailure()
            val forward=track(previous,current,pHalf,cHalf,x,y) ?: return@mapNotNull null
            val nx=x+forward.first;val ny=y+forward.second
            val back=track(current,previous,cHalf,pHalf,nx,ny) ?: return@mapNotNull null
            if(hypot(forward.first+back.first,forward.second+back.second)>1.5f || forward.third>config.maxPatchResidual) null
            else FlowPoint(x,y,nx,ny,forward.third)
        }
        if(System.nanoTime()>deadline) return budgetFailure()
        return fit(points,previous.width,previous.height)
    }
    private fun lk(a: GrayImage,b: GrayImage,x: Float,y: Float,startX: Float,startY: Float): Triple<Float,Float,Float>? {
        var u=startX;var v=startY
        if(x<4 || y<4 || x>a.width-5 || y>a.height-5) return null
        repeat(12) {
            if(x+u<4 || y+v<4 || x+u>b.width-5 || y+v>b.height-5) return null
            var xx=0f;var yy=0f;var xy=0f;var ex=0f;var ey=0f
            for(j in -3..3) for(i in -3..3) {
                val px=x+i;val py=y+j
                val gx=(a.at(px+1,py)-a.at(px-1,py))/2;val gy=(a.at(px,py+1)-a.at(px,py-1))/2
                val error=b.at(px+u,py+v)-a.at(px,py)
                xx+=gx*gx;yy+=gy*gy;xy+=gx*gy;ex+=gx*error;ey+=gy*error
            }
            val determinant=xx*yy-xy*xy
            if(determinant<1e-6f) return null
            val du=(-yy*ex+xy*ey)/determinant;val dv=(xy*ex-xx*ey)/determinant
            if(!du.isFinite() || !dv.isFinite() || abs(du)>4 || abs(dv)>4) return null
            u+=du;v+=dv
        }
        var residual=0f
        for(j in -3..3) for(i in -3..3) { val error=b.at(x+i+u,y+j+v)-a.at(x+i,y+j);residual+=error*error }
        return Triple(u,v,sqrt(residual/49))
    }
    /** Public for deterministic geometric negative cases, independently of pixel tracking. */
    fun fit(points: List<FlowPoint>,width: Int,height: Int): FlowFit {
        if(points.size<2) return FlowFit(points,0,0f,0,null,null,null,null)
        fun residual(p: FlowPoint,a: Float,b: Float,tx: Float,ty: Float)=hypot(a*p.x-b*p.y+tx-p.nx,b*p.x+a*p.y+ty-p.ny)
        var best=emptyList<FlowPoint>();var iterations=0
        loop@ for(i in points.indices) for(j in i+1 until points.size) {
            val p=points[i];val q=points[j];val x=q.x-p.x;val y=q.y-p.y;val den=x*x+y*y
            if(den<225) continue
            val nx=q.nx-p.nx;val ny=q.ny-p.ny;val a=(nx*x+ny*y)/den;val b=(ny*x-nx*y)/den
            val tx=p.nx-a*p.x+b*p.y;val ty=p.ny-b*p.x-a*p.y
            val inliers=points.filter {residual(it,a,b,tx,ty)<=config.maxFitResidual}
            if(inliers.size>best.size) best=inliers
            if(++iterations>=128) break@loop
        }
        if(best.size<2) return FlowFit(points,0,0f,0,null,null,null,null)
        val cx=best.map{it.x}.average().toFloat();val cy=best.map{it.y}.average().toFloat()
        val nx=best.map{it.nx}.average().toFloat();val ny=best.map{it.ny}.average().toFloat()
        var den=0f;var aa=0f;var bb=0f
        best.forEach {p->val x=p.x-cx;val y=p.y-cy;den+=x*x+y*y;aa+=x*(p.nx-nx)+y*(p.ny-ny);bb+=x*(p.ny-ny)-y*(p.nx-nx)}
        if(den<1e-6f) return FlowFit(points,best.size,best.size.toFloat()/points.size,0,null,null,null,null)
        val a=aa/den;val b=bb/den;val tx=nx-a*cx+b*cy;val ty=ny-b*cx-a*cy
        val coverage=best.map {(if(it.x<width/2f) 0 else 1)+(if(it.y<height/2f) 0 else 2)}.toSet().size
        return FlowFit(points,best.size,best.size.toFloat()/points.size,coverage,hypot(a,b),atan2(b,a),
            hypot(a*width/2-b*height/2+tx-width/2,b*width/2+a*height/2+ty-height/2)/min(width,height),
            best.map {residual(it,a,b,tx,ty)}.average().toFloat())
    }
}

data class WallSpatialSupport(val groundCoverage: Float?,val wallGridCells: Int?,val groundGridCells: Int?,
    val boundaryAdjacencies: Int?,val wallDepthMedian: Float?,val wallDepthQ10: Float?,val wallDepthQ90: Float?)
data class WallApproachRecord(val state: WallEvidenceState,val wallPresent: Boolean?,val possibleNear: Boolean,
    val approachingCandidate: Boolean,val confidence: Float,val abstainReason: String?,val key: SpatialFrameKey,
    val sourceAgeMs: Long,val temporalSupport: Int,val motion: MotionEvidence,val depthSupport: String,
    val flow: FlowFit?,val flowLatencyMs: Long,val automaticAlertEnabled: Boolean=false,val spatial: WallSpatialSupport?=null)

/** Uncalibrated camera expansion remains a shadow candidate. Device acceptance is mandatory. */
class WallApproachRules(private val config: WallResearchConfig=WallResearchConfig()) {
    private var session: Pair<Long,Long>?=null
    private var lastKey: SpatialFrameKey?=null
    private var candidateWindows=0
    private var presenceWindows=0
    fun reset() {session=null;lastKey=null;candidateWindows=0;presenceWindows=0}
    fun evaluate(key: SpatialFrameKey,nowMs: Long,wallSupported: Boolean,flow: FlowFit?,motion: MotionEvidence,
        depthSupport: String,latencyMs: Long=0,sceneCurrent: Boolean=true): WallApproachRecord {
        if(session!=(key.generation to key.epoch)) {reset();session=key.generation to key.epoch}
        val previous=lastKey;lastKey=key
        val age=nowMs-key.capturedAtMs
        val interval=previous?.let {key.timestampMs-it.timestampMs}
        var reason: String?=when {
            age !in 0..config.maxSourceAgeMs -> "source_stale"
            !sceneCurrent -> "source_scene_changed"
            !wallSupported -> "wall_not_supported"
            previous==null -> "need_previous_source"
            previous.width!=key.width || previous.height!=key.height || previous.rotationDegrees!=key.rotationDegrees -> "source_geometry_changed"
            interval==null || interval !in config.minPairMs..config.maxPairMs -> "frame_interval_invalid"
            !motion.synchronized || motion.quality in setOf(MotionQuality.UNKNOWN,MotionQuality.LIMITED) -> "imu_unsynchronized_or_missing"
            motion.quality!=MotionQuality.STABLE || (motion.gyroRadPerSecond ?: Float.MAX_VALUE)>config.maxGyro -> "rotation_or_shake"
            motion.intervalStartNs==null || previous.captureTimestampNs==null || motion.intervalStartNs>previous.captureTimestampNs -> "imu_window_incomplete"
            (motion.rotationMagnitudeRad ?: Float.MAX_VALUE)>config.maxRotationRad -> "rotation_explanation"
            flow?.budgetExceeded==true -> "flow_budget_exhausted"
            flow==null || flow.inliers<config.minFeatures -> "texture_or_features_insufficient"
            flow.inlierFraction<config.minInlierFraction || flow.coverage<3 -> "flow_mismatch_or_coverage"
            flow.residual==null || flow.residual>config.maxFitResidual -> "flow_fit_residual"
            flow.rotationRad==null || abs(flow.rotationRad)>config.maxRotationRad -> "visual_rotation_explanation"
            flow.centerTranslation==null || flow.centerTranslation>.035f -> "lateral_or_local_camera_motion"
            flow.scale==null || flow.scale !in config.minExpansion..config.maxExpansion -> "no_consistent_expansion"
            else -> null
        }
        if(age in 0..config.maxSourceAgeMs && sceneCurrent && wallSupported) presenceWindows++ else presenceWindows=0
        candidateWindows=if(reason==null) candidateWindows+1 else 0
        val candidate=candidateWindows>=config.confirmationWindows
        // No phone calibration, independent motion labels or verified physical translation source yet.
        if(candidate) reason="camera_translation_unverified_body_motion_unknown"
        val state=when {
            age !in 0..config.maxSourceAgeMs || !sceneCurrent -> WallEvidenceState.UNKNOWN
            !wallSupported || presenceWindows<2 -> WallEvidenceState.ABSTAIN
            reason=="no_consistent_expansion" -> WallEvidenceState.WALL_PRESENT
            else -> WallEvidenceState.ABSTAIN
        }
        return WallApproachRecord(state,if(age in 0..config.maxSourceAgeMs && sceneCurrent) wallSupported && presenceWindows>=2 else null,
            false,candidate,if(candidate) (flow?.inlierFraction ?: 0f) else 0f,reason,key,age,candidateWindows,motion,depthSupport,flow,latencyMs)
    }
}
