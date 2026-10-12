package org.sightguard.app

import kotlin.math.*

data class Vec3(val x: Float,val y: Float,val z: Float) {
    operator fun minus(v: Vec3)=Vec3(x-v.x,y-v.y,z-v.z)
    operator fun times(s: Float)=Vec3(x*s,y*s,z*s)
    fun dot(v: Vec3)=x*v.x+y*v.y+z*v.z
    fun cross(v: Vec3)=Vec3(y*v.z-z*v.y,z*v.x-x*v.z,x*v.y-y*v.x)
    fun length()=sqrt(dot(this))
    fun finite()=x.isFinite() && y.isFinite() && z.isFinite()
}
data class ArDepthPoint(val u: Float,val v: Float,val opticalZ: Float,val confidence: Float,val world: Vec3)
data class ArGeometry(val timestampNs: Long,val depthTimestampNs: Long,val receivedAtMs: Long,val rotation: Int,
    val tracking: String,val trackingFailure: String,val camera: Vec3,val forward: Vec3,val points: List<ArDepthPoint>,
    val captureElapsedMs: Long=receivedAtMs,val androidCameraTimestampNs: Long=timestampNs,val captureClockKnown: Boolean=true)
data class ArPreviewMapping(val basis: List<Float>,val widthFraction: Float,val heightFraction: Float) {
    fun point(u: Float,v: Float): Pair<Float,Float> =
        ((1-widthFraction)/2+(basis[0]+u*(basis[2]-basis[0])+v*(basis[4]-basis[0]))*widthFraction) to
        ((1-heightFraction)/2+(basis[1]+u*(basis[3]-basis[1])+v*(basis[5]-basis[1]))*heightFraction)
}
data class ArSpeechConfig(val maxAgeMs: Long=500,val maxDepthAgeMs: Long=100,val maxPairMs: Long=1600,
    val confirmations: Int=3,val minimumPoints: Int=30,val minimumConfidence: Float=.78f,
    val planeResidualM: Float=.06f,val inlierFraction: Float=.8f,val nearM: Float=1.5f,
    val approachDeltaM: Float=.15f,val cooldownMs: Long=15000,val recoveryMs: Long=2500) {
    init {
        require(maxAgeMs in 1..500 && maxDepthAgeMs in 1..500 && maxPairMs in 1..2000 && confirmations>=3 && minimumPoints>=20)
        require(minimumConfidence in .5f..1f && planeResidualM in .01f.. .1f && inlierFraction in .7f..1f && nearM in .5f..2f && approachDeltaM>=.1f && cooldownMs>=5000 && recoveryMs>0)
    }
}
data class ArPlane(val normal: Vec3,val offset: Float,val residualM: Float,val inliers: Int,val fraction: Float)
data class ArSpatialRecord(val key: SpatialFrameKey,val generatedAtMs: Long,val state: String,val reasons: List<String>,
    val wallDistanceM: Float?,val near: Boolean,val approaching: Boolean,val candidateDirection: Direction?,
    val temporalSupport: Int,val plane: ArPlane?,val geometry: ArGeometry,
    val corridorReasons: List<String>,val scope: String="EXPERIMENTAL_CAMERA_EVIDENCE_NOT_NAVIGATION")
data class ArNotice(val id: String,val text: String,val validUntilMs: Long,val direction: Direction?)

object ArGeometryRules {
    fun rotate(x: Float,y: Float,degrees: Int): Pair<Float,Float> = when(degrees) {
        0 -> x to y;90 -> 1-y to x;180 -> 1-x to 1-y;270 -> y to 1-x;else -> error("invalid rotation")
    }
    fun unrotate(x: Float,y: Float,degrees: Int)=rotate(x,y,(360-degrees)%360)
    /** Deterministic bounded RANSAC. Reject mixed surfaces instead of averaging their ranges. */
    fun plane(points: List<Vec3>,config: ArSpeechConfig): ArPlane? {
        if(points.size<config.minimumPoints || points.any {!it.finite()}) return null
        var best: ArPlane?=null
        for(i in 0 until minOf(96,points.size)) {
            val a=points[i];val b=points[(i*17+points.size/3)%points.size];val c=points[(i*31+points.size*2/3)%points.size]
            val cross=(b-a).cross(c-a);val length=cross.length()
            if(length<.015f) continue
            val normal=cross*(1/length);val offset=normal.dot(a)
            val residuals=points.map {abs(normal.dot(it)-offset)}
            val inliers=residuals.filter {it<=config.planeResidualM}
            if(inliers.size>(best?.inliers ?: 0)) best=ArPlane(normal,offset,inliers.average().toFloat(),inliers.size,inliers.size.toFloat()/points.size)
        }
        return best?.takeIf {it.inliers>=config.minimumPoints && it.fraction>=config.inlierFraction}
    }
}

/** Separate from RiskScorer. No training depth, phone acceleration integration, or world-path claims. */
class ArSpatialEvaluator(private val config: ArSpeechConfig=ArSpeechConfig()) {
    private val history=ArrayDeque<ArSpatialRecord>()
    private var previousDirection: Direction?=null
    private var directionSupport=0
    private var previousKey: SpatialFrameKey?=null
    private var previousGeometry: ArGeometry?=null
    fun reset() {history.clear();previousDirection=null;directionSupport=0;previousKey=null;previousGeometry=null}
    fun evaluate(key: SpatialFrameKey,semantics: GroundEvidence,geometry: ArGeometry,detections: List<Detection>,nowMs: Long,sceneCurrent: Boolean): ArSpatialRecord {
        val reasons=mutableListOf<String>()
        if(!geometry.captureClockKnown) reasons.add("camera_clock_unknown")
        if(key!=semantics.key || key.captureTimestampNs!=geometry.timestampNs || key.rotationDegrees!=geometry.rotation) reasons.add("source_mismatch")
        if(nowMs-key.capturedAtMs !in 0..config.maxAgeMs || nowMs-geometry.receivedAtMs !in 0..config.maxAgeMs) reasons.add("source_stale")
        if(geometry.timestampNs<=0 || geometry.depthTimestampNs<=0 || (geometry.timestampNs-geometry.depthTimestampNs)/1_000_000 !in 0..config.maxDepthAgeMs) reasons.add("depth_missing_or_stale")
        if(geometry.tracking!="TRACKING" || !geometry.camera.finite() || !geometry.forward.finite()) reasons.add("tracking_unavailable")
        if(!sceneCurrent) reasons.add("scene_changed")
        val previous=previousKey
        if(previous!=null && (previous.generation!=key.generation || previous.epoch!=key.epoch || previous.width!=key.width || previous.height!=key.height || previous.rotationDegrees!=key.rotationDegrees ||
                (geometry.timestampNs-(previousGeometry?.timestampNs ?: 0))/1_000_000 !in 1..config.maxPairMs)) reset()
        previousGeometry?.let {old ->
            if((geometry.camera-old.camera).length()>.7f || geometry.forward.dot(old.forward)<.94f) {reasons.add("pose_jump_or_rotation");reset()}
        }
        fun cell(p: ArDepthPoint)=semantics.cells.firstOrNull {p.u>=it.left && p.u<it.right && p.v>=it.top && p.v<it.bottom && it.confidence>=.6f}
        fun occupied(u: Float,v: Float)=detections.any {it.confidence>=.35f && u>=it.left-.03f && u<=it.right+.03f && v>=it.top-.03f && v<=it.bottom+.03f}
        val valid=geometry.points.filter {it.u in 0f..1f && it.v in 0f..1f && it.opticalZ in .25f..6f && it.confidence>=config.minimumConfidence && it.world.finite()}
        val wall=valid.filter {it.u in .25f.. .75f && it.v in .2f.. .9f && cell(it)?.classId==0 && !occupied(it.u,it.v)}
        val plane=ArGeometryRules.plane(wall.map {it.world},config)
        val wallReasons=mutableListOf<String>()
        if(wall.size<config.minimumPoints) wallReasons.add("wall_depth_support_insufficient")
        val coverage=wall.map {(if(it.u<.5f) 0 else 1)+(if(it.v<.55f) 0 else 2)}.toSet().size
        if(coverage<3) wallReasons.add("wall_spatial_coverage_limited")
        if(plane==null) wallReasons.add("wall_plane_unresolved")
        else if(abs(plane.normal.y)>.3f || abs(plane.normal.dot(geometry.forward))<.7f) wallReasons.add("wall_plane_not_front_facing_vertical")
        val distance=plane?.let {abs(it.normal.dot(geometry.camera)-it.offset)}?.takeIf {it in .25f..6f && reasons.isEmpty() && wallReasons.isEmpty()}
        val corridorReasons=mutableListOf<String>()
        val scores=Direction.entries.map {direction ->
            val bounds=when(direction) {Direction.LEFT -> .05f to .35f;Direction.CENTER -> .35f to .65f;Direction.RIGHT -> .65f to .95f}
            var supported=0;var unknown=0;var blocked=0
            val rows=BooleanArray(6)
            for(y in 0..5) for(x in 0..5) {
                val u=bounds.first+(x+.5f)/6*(bounds.second-bounds.first);val v=.55f+(y+.5f)/6*.4f
                val semantic=semantics.cells.firstOrNull {u>=it.left && u<it.right && v>=it.top && v<it.bottom && it.confidence>=.6f}
                val depth=valid.filter {abs(it.u-u)<.035f && abs(it.v-v)<.035f}
                when {
                    occupied(u,v) || semantic?.classId==0 -> blocked++
                    semantic==null || depth.isEmpty() -> unknown++
                    semantic.classId in SemanticEvidence.groundClassIds -> {supported++;rows[y]=true}
                    else -> blocked++
                }
            }
            val ground=valid.filter {it.u in bounds.first..bounds.second && it.v in .55f.. .95f && cell(it)?.classId in SemanticEvidence.groundClassIds && !occupied(it.u,it.v)}
            val groundPlane=ArGeometryRules.plane(ground.map {it.world},config)
            val height=groundPlane?.let {abs(it.normal.dot(geometry.camera)-it.offset)}
            val eligible=unknown<=7 && blocked<=2 && rows.all {it} && supported>=27 && groundPlane!=null && abs(groundPlane.normal.y)>.9f && height!=null && height in .7f..2f
            direction to if(eligible) supported/36f else 0f
        }.sortedByDescending {it.second}
        val candidate=if(reasons.isEmpty() && scores[0].second>0 && scores[0].second-scores[1].second>.08f) scores[0].first else null
        if(candidate==null) corridorReasons.add(if(scores.count {it.second>0}>1) "tied_candidates" else "insufficient_ground_depth_or_clearance_evidence")
        corridorReasons.add("body_clearance_and_alignment_unverified")
        directionSupport=if(candidate!=null && candidate==previousDirection) directionSupport+1 else if(candidate!=null) 1 else 0
        previousDirection=candidate
        val prior=history.lastOrNull()
        if(distance==null || prior?.plane?.let {old -> abs(old.normal.dot(plane!!.normal))<.97f || abs(old.offset-(if(old.normal.dot(plane.normal)<0) -plane.offset else plane.offset))>.12f}==true) history.clear()
        val support=if(distance!=null) history.size+1 else 0
        val first=history.firstOrNull()
        val span=first?.let {(geometry.timestampNs-it.geometry.timestampNs)/1_000_000} ?: 0
        val approach=distance!=null && first?.wallDistanceM!=null && first.plane!=null && support>=config.confirmations && span in 600..4000 &&
            geometry.forward.dot(first.geometry.forward)>.995f && first.wallDistanceM-distance>config.approachDeltaM &&
            abs(first.plane.normal.dot(geometry.camera-first.geometry.camera))>config.approachDeltaM &&
            abs((first.wallDistanceM-distance)-abs(first.plane.normal.dot(geometry.camera-first.geometry.camera)))<.1f
        val record=ArSpatialRecord(key,nowMs,if(distance!=null) "WALL_DISTANCE_ESTIMATE" else "ABSTAIN",reasons+wallReasons,
            distance,distance!=null && distance<config.nearM && support>=config.confirmations,approach,
            candidate.takeIf {directionSupport>=config.confirmations},support,plane,geometry,corridorReasons)
        if(reasons.isNotEmpty()) reset() else {
            previousKey=key;previousGeometry=geometry
            if(distance!=null) {history.addLast(record);while(history.size>=config.confirmations) history.removeFirst()}
        }
        return record
    }
}

/** Playback acknowledgement, bounded recovery and source-based deadlines, independent of formal risk. */
class ArSpeechPolicy(private val config: ArSpeechConfig=ArSpeechConfig()) {
    private val spoken=mutableSetOf<String>()
    private val pending=mutableSetOf<String>()
    private val lastSeen=mutableMapOf<String,Long>()
    private val lastStarted=mutableMapOf<String,Long>()
    private var episode=0L
    fun reset() {spoken.clear();pending.clear();lastSeen.clear();lastStarted.clear();episode++}
    fun invalidate() {spoken.clear();pending.clear();lastSeen.clear();episode++}
    fun playback(id: String,started: Boolean,nowMs: Long) {pending.remove(id);if(started) {spoken.add(id);lastStarted[id.substringAfterLast(':')]=nowMs}}
    fun notices(record: ArSpatialRecord,nowMs: Long): List<ArNotice> {
        if(nowMs-record.key.capturedAtMs !in 0 until config.maxAgeMs || record.geometry.tracking!="TRACKING") return emptyList()
        val texts=linkedMapOf<String,Pair<String,Direction?>>()
        if(record.approaching) texts["approach"]="相机可能正在靠近前方墙面，请注意。" to Direction.CENTER
        else if(record.near) texts["near"]="前方墙面可能较近，请注意。" to Direction.CENTER
        record.candidateDirection?.let {direction ->
            val label=when(direction) {Direction.LEFT -> "左侧";Direction.CENTER -> "中央";Direction.RIGHT -> "右侧"}
            texts["corridor_$direction"]="画面${label}可见地面较连续，通行条件尚未确认。" to direction
        }
        lastSeen.keys.toList().forEach {kind ->if(kind !in texts && nowMs-lastSeen.getValue(kind)>config.recoveryMs) {spoken.removeAll {it.endsWith(":$kind")};lastSeen.remove(kind)}}
        return texts.mapNotNull {(kind,pair) ->
            lastSeen[kind]=nowMs
            val id="research:${record.key.generation}:${record.key.epoch}:$episode:$kind"
            if(id in spoken || id in pending || nowMs-(lastStarted[kind] ?: -config.cooldownMs)<config.cooldownMs) null
            else ArNotice(id,pair.first,record.key.capturedAtMs+config.maxAgeMs,pair.second)
        }
    }
    fun offered(id: String) {pending.add(id)}
}
