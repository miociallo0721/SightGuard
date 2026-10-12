package org.sightguard.app

import kotlin.math.*

/** Presence labels do not imply stair direction, distance, or safety. */
enum class EnvironmentKind(val classIds: Set<Int>, val label: String, val text: String, val furniture: Boolean=false) {
    WALL(setOf(0), "墙面", "前方检测到墙面。"),
    STAIRS(setOf(53,59), "楼梯", "前方检测到楼梯。"),
    STEP(setOf(121), "台阶", "前方检测到台阶，请注意。"),
    POLE(setOf(42,93), "杆柱结构", "前方检测到杆状结构。"),
    RAILING(setOf(38,95), "栏杆或扶手", "前方检测到栏杆或扶手。"),
    TABLE(setOf(15,33,64), "桌子", "前方检测到桌子。",true),
    CHAIR(setOf(19,30,31,69,75,110), "椅子", "前方检测到椅子。",true),
    CABINET(setOf(10,35), "柜子", "前方检测到柜子。",true),
    SOFA(setOf(23), "沙发", "前方检测到沙发。",true),
    BED(setOf(7), "床", "前方检测到床。",true),
    SHELF(setOf(24), "架子", "前方检测到架子。",true),
    COUNTER(setOf(45,70), "柜台或台面", "前方检测到柜台或台面。",true),
    APPLIANCE(setOf(50), "冰箱", "前方检测到冰箱。",true),
}
data class StructureThreshold(val minFraction: Float, val minCells: Int, val confidence: Float, val confirmations: Int=2)
data class EnvironmentConfig(
    val schemaVersion: Int=1, val maxSourceAgeMs: Long=500, val sampleIntervalMs: Long=650,
    val recoveryMs: Long=4000, val cooldownMs: Long=15000, val confirmationCount: Int=2,
    val minConfirmationSpanMs: Long=250, val minConfidence: Float=.55f, val minRoiFraction: Float=.12f,
    val minCells: Int=12, val roiLeft: Float=.3f, val roiTop: Float=.2f,
    val roiRight: Float=.7f, val roiBottom: Float=.85f, val maxSceneDifference: Float=.12f,
    val thresholds: Map<EnvironmentKind,StructureThreshold> = mapOf(
        EnvironmentKind.STAIRS to StructureThreshold(.035f,8,.55f),
        EnvironmentKind.STEP to StructureThreshold(.015f,5,.65f,3),
        EnvironmentKind.POLE to StructureThreshold(.012f,6,.6f),
        EnvironmentKind.RAILING to StructureThreshold(.025f,8,.6f)),
    val automaticKinds: Set<EnvironmentKind> = EnvironmentKind.entries.filterNot { it.furniture }.toSet(),
    val useFamilyConfidence: Boolean = false,
    val preferredInputSize: Int = 256,
    val automaticModelRequirements: Map<EnvironmentKind,String> = emptyMap(),
    val resolutionFallbackAgeMs: Long = 400,
    val resolutionFallbackCount: Int = 2,
    val resolutionFallbackPrimarySlow: Int = 5,
) {
    init {
        require(schemaVersion in 1..3 && maxSourceAgeMs>0 && sampleIntervalMs>0)
        require(preferredInputSize in setOf(256,320,384))
        require(resolutionFallbackAgeMs in 1..maxSourceAgeMs && resolutionFallbackCount>=2 && resolutionFallbackPrimarySlow>0)
        require(automaticModelRequirements.values.all { it.matches(Regex("[0-9a-f]{64}")) })
        require(recoveryMs>sampleIntervalMs && cooldownMs>=0 && confirmationCount>=2)
        require(minConfirmationSpanMs>0 && minCells>0)
        require(listOf(minConfidence,minRoiFraction,maxSceneDifference,roiLeft,roiTop,roiRight,roiBottom).all { it.isFinite() && it in 0f..1f })
        require(roiLeft<roiRight && roiTop<roiBottom)
        require(thresholds.values.all { it.minFraction in 0f..1f && it.confidence in 0f..1f && it.minCells>0 && it.confirmations>=2 })
    }
    fun threshold(kind: EnvironmentKind): StructureThreshold = thresholds[kind] ?: StructureThreshold(minRoiFraction,minCells,minConfidence,confirmationCount)
}
data class EnvironmentSupport(
    val kind: EnvironmentKind, val roiFraction: Float, val confidence: Float, val cells: Int, val supported: Boolean,
    val rawClassIds: Set<Int> = emptySet(), val direction: Direction=Direction.CENTER,
    val left: Float=0f, val top: Float=0f, val right: Float=0f, val bottom: Float=0f,
    val maskKeys: Set<Int> = emptySet(), val confidenceP10: Float=0f,
    val rawConfidence: Float=confidence, val confidenceBasis: String="raw_class",
)
data class EnvironmentNotice(
    val id: String, val kind: EnvironmentKind, val sourceMs: Double, val frameIndex: Int, val validUntilMs: Long,
    val direction: Direction=Direction.CENTER, val rawClassIds: Set<Int> = emptySet(),
    val support: EnvironmentSupport? = null,
) { val text: String get() = environmentText(kind,direction) }
fun environmentText(kind: EnvironmentKind,direction: Direction): String =
    when(direction) { Direction.LEFT -> "左侧"; Direction.RIGHT -> "右侧"; else -> "前方" } + kind.text.removePrefix("前方")

/** Pixel connected components. Adjacent stairs/stairway/step share one region, preserving raw IDs. */
class SemanticStructureMeasure(private val config: EnvironmentConfig) {
    private fun family(id: Int): EnvironmentKind? = EnvironmentTaxonomy.family(id)
    private fun confidenceThreshold(id: Int): Float = config.threshold(EnvironmentTaxonomy.kindForClass(id) ?: EnvironmentKind.WALL).confidence
    private fun confidence(cell: SemanticCell): Float {
        if(!config.useFamilyConfidence || EnvironmentTaxonomy.kindForClass(cell.classId)?.furniture!=true) return cell.confidence
        val value=cell.familyConfidence ?: return cell.confidence
        return if(value.isFinite() && value in cell.confidence..1f) value else 0f
    }
    fun components(cells: List<SemanticCell>): List<EnvironmentSupport> {
        val valid=cells.filter { family(it.classId)!=null && it.confidence.isFinite() && it.confidence in 0f..1f && confidence(it) in confidenceThreshold(it.classId)..1f &&
            listOf(it.left,it.top,it.right,it.bottom).all { v->v.isFinite() && v in 0f..1f } && it.left<it.right && it.top<it.bottom }
        if(valid.isEmpty()) return emptyList()
        val widths=valid.map { it.right-it.left }.sorted(); val heights=valid.map { it.bottom-it.top }.sorted()
        val dx=widths[widths.size/2]; val dy=heights[heights.size/2]
        fun index(c: SemanticCell) = (c.left/dx).roundToInt() to (c.top/dy).roundToInt()
        val grid=valid.groupBy { index(it) }; val remaining=valid.toMutableSet(); val result=mutableListOf<EnvironmentSupport>()
        while(remaining.isNotEmpty()) {
            val first=remaining.first(); remaining.remove(first)
            val queue=ArrayDeque<SemanticCell>(); queue.add(first); val component=mutableListOf<SemanticCell>()
            while(queue.isNotEmpty()) {
                val cell=queue.removeFirst(); component.add(cell); val (x,y)=index(cell)
                for(position in listOf(x-1 to y,x+1 to y,x to y-1,x to y+1)) grid[position]?.forEach { other ->
                    val horizontal=abs(cell.right-other.left)<1e-4f || abs(other.right-cell.left)<1e-4f
                    val vertical=abs(cell.bottom-other.top)<1e-4f || abs(other.bottom-cell.top)<1e-4f
                    val touches=(horizontal && min(cell.bottom,other.bottom)>max(cell.top,other.top)) ||
                        (vertical && min(cell.right,other.right)>max(cell.left,other.left))
                    if(family(other.classId)==family(first.classId) && touches && remaining.remove(other)) queue.add(other)
                }
            }
            val ids=component.map { it.classId }.toSet()
            val kind=if(family(first.classId)==EnvironmentKind.STAIRS && ids.none {it==53 || it==59}) EnvironmentKind.STEP else family(first.classId)!!
            val threshold=config.threshold(kind)
            val attentionLeft=if(kind.furniture || kind in setOf(EnvironmentKind.POLE,EnvironmentKind.RAILING)) .08f else config.roiLeft
            val attentionRight=if(kind.furniture || kind in setOf(EnvironmentKind.POLE,EnvironmentKind.RAILING)) .92f else config.roiRight
            val accepted=component.filter { confidence(it)>=threshold.confidence }
            var area=0f; var weighted=0f; var rawWeighted=0f; var count=0; var centerSum=0f
            accepted.forEach { c ->
                val overlap=max(0f,min(attentionRight,c.right)-max(attentionLeft,c.left))*max(0f,min(config.roiBottom,c.bottom)-max(config.roiTop,c.top))
                if(overlap>0) { area+=overlap; weighted+=overlap*confidence(c); rawWeighted+=overlap*c.confidence; centerSum+=overlap*(c.left+c.right)/2; count++ }
            }
            val fraction=(area/((attentionRight-attentionLeft)*(config.roiBottom-config.roiTop))).coerceAtMost(1f)
            val center=if(area>0) centerSum/area else .5f
            val direction=if(kind==EnvironmentKind.WALL) Direction.CENTER else when { center<1f/3->Direction.LEFT; center>2f/3->Direction.RIGHT; else->Direction.CENTER }
            val confidences=accepted.map { confidence(it) }.sorted()
            result+=EnvironmentSupport(kind,fraction,if(area>0) weighted/area else 0f,count,
                fraction>=threshold.minFraction && count>=threshold.minCells,ids,direction,
                component.minOf { it.left },component.minOf { it.top },component.maxOf { it.right },component.maxOf { it.bottom },
                accepted.map { ((it.top+it.bottom)*32).toInt().coerceIn(0,63)*64+((it.left+it.right)*32).toInt().coerceIn(0,63) }.toSet(),
                confidences.getOrElse((confidences.size*.1f).toInt()) { 0f },
                if(area>0) rawWeighted/area else 0f,
                if(kind.furniture && config.useFamilyConfidence && accepted.any {it.familyConfidence!=null}) "furniture_family_softmax_sum" else "raw_class")
        }
        return result.sortedByDescending { it.roiFraction }
    }
}

/** Source-time confirmation and per-region identity. Elapsed time gates actual playback freshness. */
class SemanticEnvironmentEngine(val config: EnvironmentConfig=EnvironmentConfig()) {
    private data class Track(val serial: Long,var support: EnvironmentSupport,var first: Double,var last: Double,
        var lastCapture: Long,var count: Int=1,var id: String?=null,var spoken: Boolean=false,var notice: EnvironmentNotice?=null,
        var heardByDetection: Boolean=false)
    private val tracks=mutableListOf<Track>()
    private val recentlySpoken=mutableListOf<Pair<EnvironmentSupport,Double>>()
    private val measure=SemanticStructureMeasure(config)
    private var session: Pair<Long,Long>?=null
    private var lastSource=Double.NEGATIVE_INFINITY
    private var sequence=0L
    private var modelSha: String?=null
    var status: String="环境：等待有效语义观测"; private set
    var silenceReason: String="当前没有满足提醒条件的证据。"; private set
    var supports: List<EnvironmentSupport> = emptyList(); private set
    val approachEnabled: Boolean=false
    fun reset() {
        tracks.clear(); recentlySpoken.clear(); session=null; lastSource=Double.NEGATIVE_INFINITY; supports=emptyList(); modelSha=null
        status="环境：等待有效语义观测"; silenceReason="当前没有满足提醒条件的证据。"
    }
    /** Compatibility summary: one entry per kind, confirmation still consumes individual regions. */
    fun measure(cells: List<SemanticCell>): List<EnvironmentSupport> {
        val parts=measure.components(cells)
        return EnvironmentKind.entries.map { kind -> parts.filter { it.kind==kind }.maxByOrNull { it.roiFraction } ?: EnvironmentSupport(kind,0f,0f,0,false) }
    }
    private fun compatible(a: EnvironmentSupport,b: EnvironmentSupport): Boolean = a.kind==b.kind ||
        (a.kind in setOf(EnvironmentKind.STAIRS,EnvironmentKind.STEP) && b.kind in setOf(EnvironmentKind.STAIRS,EnvironmentKind.STEP))
    private fun overlap(a: EnvironmentSupport,b: EnvironmentSupport): Float = if(!compatible(a,b) || a.maskKeys.isEmpty() || b.maskKeys.isEmpty()) 0f
        else a.maskKeys.intersect(b.maskKeys).size.toFloat()/min(a.maskKeys.size,b.maskKeys.size)
    fun observe(evidence: GroundEvidence,nowMs: Long,sceneCurrent: Boolean,rapidMotion: Boolean): List<EnvironmentNotice> {
        val key=evidence.key
        if(session!=(key.generation to key.epoch)) { reset(); session=key.generation to key.epoch }
        if(modelSha!=evidence.modelSha) {
            // New model must establish its own observations; keep already-heard identities.
            tracks.forEach {it.notice=null;it.count=0;it.first=key.timestampMs}
            modelSha=evidence.modelSha
        }
        val eligibleKinds=config.automaticKinds.filter {config.automaticModelRequirements[it]?.let {sha->sha==evidence.modelSha} ?: true}.toSet()
        if(!key.timestampMs.isFinite() || key.timestampMs<=lastSource) return pending(nowMs)
        lastSource=key.timestampMs
        if(nowMs-key.capturedAtMs !in 0..config.maxSourceAgeMs || evidence.completedAtMs !in key.capturedAtMs..nowMs || evidence.latencyMs<0 || !sceneCurrent || rapidMotion) {
            invalidate(nowMs,"环境：证据过期或画面变化，等待重新确认"); return emptyList()
        }
        expire(nowMs); supports=measure.components(evidence.cells); val used=mutableSetOf<Long>()
        supports.filter { it.supported }.forEach { support ->
            val old=tracks.filter { it.serial !in used && overlap(it.support,support)>=.35f }.maxByOrNull { overlap(it.support,support) }
            val track=if(old==null) Track(++sequence,support,key.timestampMs,key.timestampMs,key.capturedAtMs).also { tracks.add(it) }
                else old.apply {
                    if(count==0) first=key.timestampMs
                    count++; last=key.timestampMs; lastCapture=key.capturedAtMs
                    this.support=if(this.support.kind==EnvironmentKind.STAIRS && support.kind==EnvironmentKind.STEP) support.copy(kind=EnvironmentKind.STAIRS,rawClassIds=this.support.rawClassIds+support.rawClassIds)
                        else support.copy(rawClassIds=this.support.rawClassIds+support.rawClassIds)
                }
            used.add(track.serial)
            if(track.support.kind in eligibleKinds && track.count>=config.threshold(track.support.kind).confirmations && key.timestampMs-track.first>=config.minConfirmationSpanMs) {
                if(track.id==null) track.id="semantic:${key.generation}:${key.epoch}:${track.support.kind}:${track.serial}"
                track.notice=EnvironmentNotice(track.id!!,track.support.kind,key.timestampMs,key.frameIndex,key.capturedAtMs+config.maxSourceAgeMs,track.support.direction,track.support.rawClassIds,track.support)
            }
        }
        tracks.filter { it.serial !in used }.forEach { it.notice=null; if(it.id==null) it.count=0 }
        val current=tracks.filter { it.serial in used && it.notice!=null }
        status=when {
            current.isNotEmpty()->current.joinToString("；") { "${it.support.kind.label}：已确认存在" }
            used.isNotEmpty() && supports.none {it.supported && it.kind in eligibleKinds}->"环境：该类别提醒能力尚未验证"
            used.isNotEmpty()->"环境：连续观测确认中"
            tracks.isNotEmpty()->"环境：暂时丢失，有限恢复中"
            else->"环境：当前未确认"
        }
        silenceReason=when {
            current.any { !it.spoken && !it.heardByDetection }->"等待音频调度。"
            current.any { it.heardByDetection }->"同一区域已由目标提醒播报。"
            current.isNotEmpty()->"此目标已提醒过。"
            used.isNotEmpty() && supports.none {it.supported && it.kind in eligibleKinds}->"该类别提醒能力尚未验证。"
            used.isNotEmpty()->"正在连续确认。"
            else->"当前没有满足提醒条件的证据。"
        }
        return pending(nowMs)
    }
    fun currentNotices(nowMs: Long): List<EnvironmentNotice> {
        expire(nowMs); return tracks.mapNotNull { it.notice?.takeIf { n->nowMs<n.validUntilMs } }
    }
    fun pending(nowMs: Long): List<EnvironmentNotice> {
        expire(nowMs)
        return tracks.mapNotNull { t->t.notice?.takeIf { n-> !t.spoken && !t.heardByDetection && nowMs<n.validUntilMs &&
            recentlySpoken.none { (s,time)->overlap(s,t.support)>=.35f && n.sourceMs-time<config.cooldownMs } } }
            .sortedWith(compareBy<EnvironmentNotice> { when(it.kind) { EnvironmentKind.STAIRS,EnvironmentKind.STEP->0; EnvironmentKind.POLE->1; EnvironmentKind.RAILING->2; else->3 } }.thenBy { it.sourceMs })
    }
    fun validIds(nowMs: Long): Set<String> = currentNotices(nowMs).map { it.id }.toSet()
    fun playbackStarted(id: String) { tracks.find { it.id==id }?.let { it.spoken=true; recentlySpoken.add(it.support to it.last); silenceReason="此目标已提醒过。" } }
    fun acknowledgedByDetection(id: String) { tracks.find { it.id==id }?.let {
        it.heardByDetection=true; recentlySpoken.add(it.support to it.last); silenceReason="同一区域已由目标提醒播报。"
    } }
    fun invalidate(nowMs: Long,reason: String) {
        tracks.forEach { it.notice=null; it.count=0; it.first=lastSource }; supports=emptyList(); expire(nowMs); status=reason; silenceReason="感知结果已过期。"
    }
    private fun expire(nowMs: Long) {
        val removed=tracks.removeAll { nowMs-it.lastCapture>config.recoveryMs }
        recentlySpoken.removeAll { lastSource-it.second>config.cooldownMs }
        if(removed && tracks.isEmpty()) status="环境：事件已结束"
        else if(tracks.isNotEmpty() && tracks.all { nowMs-it.lastCapture>=config.maxSourceAgeMs }) { status="环境：等待新鲜证据"; silenceReason="感知结果已过期。" }
    }
}
object SemanticSceneContinuity {
    fun matches(source: FloatArray,latest: FloatArray,maximum: Float): Boolean =
        source.size==latest.size && source.isNotEmpty() && source.indices.all { source[it].isFinite() && latest[it].isFinite() } &&
            source.indices.sumOf { abs(source[it]-latest[it]).toDouble() }/source.size<=maximum
}
