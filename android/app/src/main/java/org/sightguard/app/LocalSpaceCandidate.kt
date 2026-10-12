package org.sightguard.app

/** Image directions only. No pose, physical clearance, route or safety inference. */
data class LocalSpaceConfig(
    val maxAgeMs: Long = 1200, val maxGapMs: Long = 4000,
    val confirmations: Int = 3, val cooldownMs: Long = 20000,
    val minGround: Float = .75f, val maxOccupancy: Float = .08f,
    val minConnected: Float = .65f, val tieMargin: Float = .15f,
) {
    init {
        require(maxAgeMs>0 && maxGapMs>0 && confirmations>=2 && cooldownMs>=0)
        require(listOf(minGround,maxOccupancy,minConnected,tieMargin).all {it.isFinite() && it in 0f..1f})
    }
}
data class LocalSpaceBand(val direction: Direction, val connectedFraction: Float, val reachesTop: Boolean,
    val unknownFraction: Float, val eligible: Boolean)
data class LocalSpaceRecord(val key: SpatialFrameKey, val status: String, val reason: String,
    val bands: List<LocalSpaceBand> = emptyList(), val direction: Direction? = null,
    val support: Int = 0, val validUntilMs: Long = 0, val eventId: String? = null,
    val supportedCells: List<GridCell> = emptyList()) {
    val text: String? get() = direction?.let {
        "画面${when(it) {Direction.LEFT->"左侧";Direction.CENTER->"中央";Direction.RIGHT->"右侧"}}可见地面较连续，通行条件尚未确认。"
    }
}

/** Reuses the 15x15 fusion contract. Unknown cells are never traversed. */
class LocalSpaceCandidateEvaluator(val config: LocalSpaceConfig = LocalSpaceConfig()) {
    private var previous: SpatialFrameKey? = null
    private var best: Direction? = null
    private var support = 0
    private var serial = 0L
    private var activeId: String? = null
    private var spokenId: String? = null
    private var lastSpokenAt = Long.MIN_VALUE / 2
    var current: LocalSpaceRecord? = null; private set
    fun reset() { invalidate(); previous=null; spokenId=null; lastSpokenAt=Long.MIN_VALUE/2 }
    fun invalidate() { best=null; support=0; activeId=null; current=null }
    fun playbackStarted(id: String, now: Long) { if(id==activeId) {spokenId=id;lastSpokenAt=now} }
    /** Primary-chain detections can revoke a candidate before another semantic inference. */
    fun obstructed(detections: List<Detection>): Boolean = current?.supportedCells?.any {c ->
        detections.any {d -> d.confidence.isFinite() && d.confidence>=Detector.MIN_SCORE &&
            maxOf(0f,minOf(c.right,d.right)-maxOf(c.left,d.left))*maxOf(0f,minOf(c.bottom,d.bottom)-maxOf(c.top,d.top)) /
            ((c.right-c.left)*(c.bottom-c.top)) > config.maxOccupancy }
    }==true
    fun pending(now: Long): LocalSpaceRecord? = current?.takeIf {
        it.status=="SUPPORTED_CANDIDATE" && now<it.validUntilMs && it.eventId!=spokenId && now-lastSpokenAt>=config.cooldownMs
    }
    fun observe(grid: SpatialGridEvidence, semantics: GroundEvidence, now: Long, sceneCurrent: Boolean): LocalSpaceRecord {
        val key=grid.key
        fun reject(reason: String): LocalSpaceRecord {
            invalidate(); previous=key
            return LocalSpaceRecord(key,"ABSTAIN",reason).also {current=it}
        }
        if(semantics.key!=key || grid.failures.any {!it.startsWith("depth:missing")}) return reject("source_mismatch_or_missing")
        if(semantics.modelSha !in listOf(SpatialDetector.MODEL_SHA,SegFormerModels.DETAIL_SHA,SegFormerModels.BALANCED_SHA) || semantics.cells.isEmpty()) return reject("unverified_semantic_source")
        if(!sceneCurrent) return reject("scene_changed")
        if(now-key.capturedAtMs !in 0 until config.maxAgeMs) return reject("source_expired")
        if(grid.motion.quality==MotionQuality.RAPID || grid.motion.quality==MotionQuality.LIMITED) return reject("motion_limited")
        if(!key.timestampMs.isFinite()) return reject("invalid_timestamp")
        previous?.let { p ->
            if(p.generation!=key.generation || p.epoch!=key.epoch || p.width!=key.width || p.height!=key.height || p.rotationDegrees!=key.rotationDegrees ||
                key.frameIndex<=p.frameIndex || key.timestampMs-p.timestampMs<=0 || key.timestampMs-p.timestampMs>config.maxGapMs) invalidate()
        }
        // Replayed/duplicate frames never accumulate confirmation.
        if(previous?.let {it.generation==key.generation && it.epoch==key.epoch && key.frameIndex<=it.frameIndex}==true) return reject("duplicate_or_reversed_frame")
        previous=key
        if(grid.cells.size!=225 || grid.cells.map {it.row to it.col}.toSet().size!=225 || grid.cells.any {it.row !in 0..14 || it.col !in 0..14}) return reject("invalid_grid")
        val byPosition=grid.cells.associateBy {it.row to it.col}
        val hazards=semantics.cells.filter {it.classId in HAZARDS && it.confidence>=.35f}
        val connectedByDirection=mutableMapOf<Direction,List<GridCell>>()
        val bands=Direction.entries.map { direction ->
            val first=when(direction) {Direction.LEFT->0;Direction.CENTER->5;Direction.RIGHT->10}
            val cells=(8..14).flatMap {r -> (first until first+5).map {c -> byPosition.getValue(r to c)}}
            fun accepted(c: GridCell): Boolean {
                if(c.state!="GROUND" || c.ground?.let {it.isFinite() && it>=config.minGround}!=true || c.occupancy?.let {it.isFinite() && it<=config.maxOccupancy}!=true) return false
                if((c.wall ?: 1f)>.05f || (c.semanticObjects ?: 0f)>.05f) return false
                // Water/stairs/steps or railing intersecting a band veto that cell.
                if(hazards.any {it.left<c.right && it.right>c.left && it.top<c.bottom && it.bottom>c.top}) return false
                return true
            }
            val allowed=cells.filter(::accepted).map {it.row to it.col}.toSet()
            val visited=mutableSetOf<Pair<Int,Int>>()
            val queue=ArrayDeque<Pair<Int,Int>>()
            allowed.filter {it.first==14}.forEach {visited.add(it);queue.add(it)}
            while(queue.isNotEmpty()) {
                val (r,c)=queue.removeFirst()
                for(next in listOf(r-1 to c,r+1 to c,r to c-1,r to c+1)) if(next in allowed && visited.add(next)) queue.add(next)
            }
            val fraction=visited.size/35f
            val reaches=(8..14).all {r -> visited.count {it.first==r}>=2}
            connectedByDirection[direction]=cells.filter {(it.row to it.col) in visited}
            LocalSpaceBand(direction,fraction,reaches,cells.count {it.state=="UNKNOWN" || it.state=="CONFLICT"}/35f,reaches && fraction>=config.minConnected)
        }
        val ranked=bands.filter {it.eligible}.sortedByDescending {it.connectedFraction}
        val winner=ranked.firstOrNull()?.takeIf {top -> bands.filter {it.direction!=top.direction}.all {top.connectedFraction-it.connectedFraction>=config.tieMargin}}
        if(winner==null) {
            invalidate()
            return LocalSpaceRecord(key,if(ranked.isEmpty()) "INSUFFICIENT_EVIDENCE" else "TIED","no_unique_connected_ground_band",bands).also {current=it}
        }
        if(best!=winner.direction) {best=winner.direction;support=0;activeId=null}
        support++
        if(support>=config.confirmations && activeId==null) activeId="local-space:${key.generation}:${key.epoch}:${++serial}"
        return LocalSpaceRecord(key,if(activeId!=null) "SUPPORTED_CANDIDATE" else "CONFIRMING",
            "image_only_pose_body_clearance_unknown;motion=${grid.motion.quality}",bands,winner.direction,support,key.capturedAtMs+config.maxAgeMs,activeId,
            connectedByDirection.getValue(winner.direction)).also {current=it}
    }
    companion object { private val HAZARDS=setOf(21,38,53,59,95,96,121) }
}
