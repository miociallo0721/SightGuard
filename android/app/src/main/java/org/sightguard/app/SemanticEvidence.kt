package org.sightguard.app

/** Scene labels are image semantics, never obstacle, distance, or safety assertions. */
enum class SemanticGroup(val displayName: String) {
    WALL("墙面"), BUILDING("建筑"), SKY("天空"), FLOOR("地板"), ROAD("道路"), SIDEWALK("人行道"),
    TREE_UPPER("上部树木/树冠语义"), TREE_LOWER("下部树木语义"), GRASS("草地"), PLANT("植物"),
    DOOR("门"), STAIRS("楼梯"), STAIRWAY("楼梯通道"), STEP("台阶"), WATER("水面"), STATIC("静态物体"), OTHER("其他语义")
}

/** Per-class/group aggregate. Bounds enclose its cells and may include disconnected gaps. */
data class SemanticRegion(
    val group: SemanticGroup, val classId: Int, val areaFraction: Float,
    val confidence: Float, val cellCount: Int,
    val left: Float, val top: Float, val right: Float, val bottom: Float,
    val direction: Direction = Direction.CENTER,
)

data class SemanticSourceQuality(val current: Boolean = true, val readable: Boolean = true, val detail: String = "", val known: Boolean = true)
data class SemanticConfig(val minConfidence: Float = .35f, val minAreaFraction: Float = .003f, val treeUpperBoundary: Float = .5f, val maxOverviewItems: Int = 5) {
    init {
        require(minConfidence.isFinite() && minConfidence in 0f..1f)
        require(minAreaFraction.isFinite() && minAreaFraction in 0f..1f)
        require(treeUpperBoundary.isFinite() && treeUpperBoundary in 0f..1f)
        require(maxOverviewItems > 0)
    }
}
data class SemanticSummary(val regions: List<SemanticRegion>, val overview: String, val reason: String = "")

/** Pure deterministic aggregation of nonoverlapping segmentation cells; no alert state. */
object SemanticEvidence {
    val groundClassIds: Set<Int> = setOf(3, 6, 11)
    val displayedClassIds: Set<Int> = setOf(0, 1, 2, 3, 4, 6, 9, 10, 11, 14, 15, 17, 19, 21, 32, 53, 59, 93, 121)
    private fun direction(centerX: Float): Direction = when { centerX < 1f / 3f -> Direction.LEFT; centerX > 2f / 3f -> Direction.RIGHT; else -> Direction.CENTER }
    private fun directionText(direction: Direction): String = when (direction) { Direction.LEFT -> "左侧"; Direction.CENTER -> "前方"; Direction.RIGHT -> "右侧" }
    private fun sceneName(region: SemanticRegion): String = when (region.classId) {
        10 -> "柜子"; 15 -> "桌子"; 19 -> "椅子"; 32 -> "围栏"; 93 -> "杆柱"
        else -> region.group.displayName
    }

    fun group(classId: Int, centerY: Float, config: SemanticConfig = SemanticConfig()): SemanticGroup = when (classId) {
        0 -> SemanticGroup.WALL; 1 -> SemanticGroup.BUILDING; 2 -> SemanticGroup.SKY
        3 -> SemanticGroup.FLOOR; 4 -> if (centerY < config.treeUpperBoundary) SemanticGroup.TREE_UPPER else SemanticGroup.TREE_LOWER
        6 -> SemanticGroup.ROAD; 9 -> SemanticGroup.GRASS; 11 -> SemanticGroup.SIDEWALK
        14 -> SemanticGroup.DOOR; 17 -> SemanticGroup.PLANT; 21 -> SemanticGroup.WATER
        53 -> SemanticGroup.STAIRS; 59 -> SemanticGroup.STAIRWAY; 121 -> SemanticGroup.STEP
        10, 15, 19, 32, 93 -> SemanticGroup.STATIC
        else -> SemanticGroup.OTHER
    }

    fun summarize(cells: List<SemanticCell>, detections: List<Detection>, quality: SemanticSourceQuality, config: SemanticConfig = SemanticConfig()): SemanticSummary {
        if (!quality.current || !quality.readable || !quality.known) return SemanticSummary(emptyList(), "当前环境信息不足。", quality.detail.ifBlank { "来源过期、不可读或质量未知" })
        val valid = cells.filter {
            it.classId in displayedClassIds && it.confidence.isFinite() && it.confidence in config.minConfidence..1f &&
                listOf(it.left, it.top, it.right, it.bottom).all { v -> v.isFinite() && v in 0f..1f } && it.right > it.left && it.bottom > it.top
        }.sortedWith(compareBy<SemanticCell> { it.classId }.thenBy { it.top }.thenBy { it.left }.thenBy { it.bottom }.thenBy { it.right }.thenBy { it.confidence })
        val regions = valid.groupBy { Triple(it.classId, group(it.classId, (it.top + it.bottom) / 2f, config), direction((it.left + it.right) / 2f)) }.map { (key, items) ->
            val area = items.sumOf { ((it.right - it.left) * (it.bottom - it.top)).toDouble() }.toFloat()
            val confidence = items.sumOf { (it.confidence * (it.right - it.left) * (it.bottom - it.top)).toDouble() }.toFloat() / area
            SemanticRegion(key.second, key.first, area, confidence, items.size, items.minOf { it.left }, items.minOf { it.top }, items.maxOf { it.right }, items.maxOf { it.bottom }, key.third)
        }.filter { it.areaFraction >= config.minAreaFraction }.sortedWith(compareByDescending<SemanticRegion> { it.areaFraction }.thenBy { it.classId }.thenBy { it.group.ordinal }.thenBy { it.direction.ordinal })
        val scene = regions.take(config.maxOverviewItems).map { directionText(it.direction) + sceneName(it) }
        val objects = detections.filter {
            it.classId in Labels.coco80.indices && it.confidence.isFinite() && it.confidence in config.minConfidence..1f &&
                listOf(it.left, it.top, it.right, it.bottom).all { v -> v.isFinite() && v in 0f..1f } && it.right > it.left && it.bottom > it.top
        }.map { it.classId to direction((it.left + it.right) / 2f) }.distinct().sortedWith(compareBy<Pair<Int, Direction>> { it.first }.thenBy { it.second.ordinal }).take(config.maxOverviewItems).map { directionText(it.second) + Labels.coco80[it.first] }
        val parts = listOfNotNull(scene.takeIf { it.isNotEmpty() }?.joinToString("、"), objects.takeIf { it.isNotEmpty() }?.let { "检测到" + it.joinToString("、") })
        return if (parts.isEmpty()) SemanticSummary(regions, "当前环境信息不足。", "缺少达到阈值的语义和检测证据")
        else SemanticSummary(regions, "环境概览：" + parts.joinToString("；") + " · 仅视觉语义")
    }
}
