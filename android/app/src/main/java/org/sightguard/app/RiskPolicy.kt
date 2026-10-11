package org.sightguard.app

/** A detector result in normalized coordinates of the upright analysis image. */
data class Detection(
    val classId: Int,
    val confidence: Float,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
)

enum class Direction { LEFT, CENTER, RIGHT }

data class RiskAlert(val text: String, val important: Boolean, val eventId: Long)

/** Chinese COCO-80 labels, indexed by the detector's standard COCO class id. */
object Labels {
    val coco80: List<String> = listOf(
        "行人", "自行车", "汽车", "摩托车", "飞机", "公交车", "火车", "卡车", "船", "红绿灯",
        "消防栓", "停止标志", "停车计时器", "长椅", "鸟", "猫", "狗", "马", "羊", "牛",
        "大象", "熊", "斑马", "长颈鹿", "背包", "雨伞", "手提包", "领带", "行李箱", "飞盘",
        "滑雪板", "单板滑雪板", "球", "风筝", "棒球棒", "棒球手套", "滑板", "冲浪板", "网球拍", "瓶子",
        "酒杯", "杯子", "叉子", "刀", "勺子", "碗", "香蕉", "苹果", "三明治", "橙子",
        "西兰花", "胡萝卜", "热狗", "披萨", "甜甜圈", "蛋糕", "椅子", "沙发", "盆栽", "床",
        "餐桌", "马桶", "电视", "笔记本电脑", "鼠标", "遥控器", "键盘", "手机", "微波炉", "烤箱",
        "烤面包机", "水槽", "冰箱", "书", "时钟", "花瓶", "剪刀", "泰迪熊", "吹风机", "牙刷",
    )
}

/**
 * Small temporal filter for detector boxes. Center/size are image-plane proxies only;
 * this policy does not estimate distance, collision, or safety.
 */
class RiskPolicy {
    private data class Track(
        val eventId: Long,
        var classId: Int,
        var direction: Direction,
        var centerX: Float,
        var centerY: Float,
        var width: Float,
        var height: Float,
        var confidence: Float,
        var firstSeenMs: Long,
        var lastSeenMs: Long,
        var consecutiveObservations: Int,
        var alerted: Boolean = false,
        var importantUpgradeSent: Boolean = false,
    )

    private var tracks = mutableListOf<Track>()
    private var nextEventId = 1L
    private var lastTimestampMs: Long? = null
    private var lastAlertMs: Long? = null
    private var lastAlertImportant = false

    fun update(detections: List<Detection>, timestampMs: Long, allowAlerts: Boolean = true): RiskAlert? {
        checkTime(timestampMs)
        expireInternal(timestampMs)

        val valid = detections.asSequence()
            .take(MAX_INPUT_DETECTIONS)
            .filter(::isUsable)
            .sortedByDescending { it.confidence }
            .take(MAX_DETECTIONS)
            .toList()
        val frameDetections = deduplicateFrameDetections(valid)
        val seenEventIds = mutableSetOf<Long>()
        for (detection in frameDetections) {
            val geometry = geometry(detection)
            val match = tracks.asSequence()
                .filter { track ->
                    track.eventId !in seenEventIds &&
                        if (timestampMs - track.lastSeenMs <= LOSS_GRACE_MS) {
                            compatible(track, detection, geometry)
                        } else {
                            compatibleRecent(track, detection, geometry)
                        }
                }
                .map { it to associationCost(it, geometry) }
                .minByOrNull { it.second }
                ?.first
            if (match == null) {
                if (tracks.size >= MAX_TRACKS) {
                    tracks.minByOrNull { it.lastSeenMs }?.let(tracks::remove)
                }
                val track = Track(
                    eventId = nextEventId++,
                    classId = detection.classId,
                    direction = directionOf(geometry.centerX),
                    centerX = geometry.centerX,
                    centerY = geometry.centerY,
                    width = geometry.width,
                    height = geometry.height,
                    confidence = detection.confidence,
                    firstSeenMs = timestampMs,
                    lastSeenMs = timestampMs,
                    consecutiveObservations = 1,
                )
                tracks.add(track)
                seenEventIds.add(track.eventId)
            } else {
                val gap = timestampMs - match.lastSeenMs
                match.consecutiveObservations = if (gap <= CONFIRMATION_MAX_GAP_MS) {
                    match.consecutiveObservations + 1
                } else {
                    1
                }
                match.classId = detection.classId
                match.direction = stableDirection(match.direction, geometry.centerX)
                match.centerX = geometry.centerX
                match.centerY = geometry.centerY
                match.width = geometry.width
                match.height = geometry.height
                match.confidence = detection.confidence
                match.lastSeenMs = timestampMs
                seenEventIds.add(match.eventId)
            }
        }

        // A not-yet-confirmed candidate must be observed on adjacent processed frames.
        tracks.filter { it.eventId !in seenEventIds && !it.alerted }
            .forEach { it.consecutiveObservations = 0 }

        val eligible = tracks.asSequence()
            .filter {
                it.eventId in seenEventIds && it.consecutiveObservations >= CONFIRMATION_OBSERVATIONS &&
                    (!it.alerted || (isImportant(it) && !it.importantUpgradeSent))
            }
            .sortedWith(
                compareByDescending<Track> { isImportant(it) }
                    .thenBy { kotlin.math.abs(it.centerX - IMAGE_CENTER_X) }
                    .thenByDescending { it.confidence },
            )
            .firstOrNull()

        val alert = eligible?.takeIf { allowAlerts && mayEmit(timestampMs, isImportant(it)) }?.let { track ->
            val isUpgrade = track.alerted
            if (isUpgrade) {
                track.importantUpgradeSent = true
            } else {
                track.alerted = true
                track.importantUpgradeSent = isImportant(track)
            }
            lastAlertMs = timestampMs
            lastAlertImportant = isImportant(track)
            val label = Labels.coco80.getOrNull(track.classId) ?: "目标"
            val text = if (isUpgrade) {
                "${track.direction.label()}有${label}，请注意"
            } else {
                "${track.direction.label()}有${label}"
            }
            RiskAlert(text, lastAlertImportant, track.eventId)
        }
        lastTimestampMs = timestampMs
        return alert
    }

    /** Drop candidates and recovered events whose short-loss window has elapsed. */
    fun expire(timestampMs: Long) {
        checkTime(timestampMs)
        expireInternal(timestampMs)
        lastTimestampMs = timestampMs
    }

    /** Start a fresh camera session; IDs and cooldown are scoped to one session. */
    fun reset() {
        tracks.clear()
        nextEventId = 1L
        lastTimestampMs = null
        lastAlertMs = null
        lastAlertImportant = false
    }

    private fun checkTime(timestampMs: Long) {
        require(timestampMs >= 0L) { "timestampMs must be non-negative" }
        val previous = lastTimestampMs
        require(previous == null || timestampMs >= previous) { "timestampMs must not move backwards" }
    }

    private fun expireInternal(timestampMs: Long) {
        tracks.removeAll { timestampMs - it.lastSeenMs > REIDENTIFICATION_WINDOW_MS }
        if (tracks.isEmpty() && lastAlertMs != null && timestampMs - lastAlertMs!! > COOLDOWN_MS) {
            // Keep no long-lived alert history after its operational window.
            lastAlertMs = null
            lastAlertImportant = false
        }
    }

    private fun mayEmit(timestampMs: Long, important: Boolean): Boolean {
        val last = lastAlertMs ?: return true
        return timestampMs - last >= COOLDOWN_MS || (important && !lastAlertImportant)
    }

    private fun isUsable(d: Detection): Boolean =
        d.classId in Labels.coco80.indices &&
            d.confidence.isFinite() && d.confidence in MIN_CONFIDENCE..1f &&
            d.left.isFinite() && d.top.isFinite() && d.right.isFinite() && d.bottom.isFinite() &&
            d.left >= 0f && d.top >= 0f && d.right <= 1f && d.bottom <= 1f &&
            d.right > d.left && d.bottom > d.top

    private data class Geometry(val centerX: Float, val centerY: Float, val width: Float, val height: Float)

    private fun geometry(d: Detection) = Geometry(
        (d.left + d.right) / 2f,
        (d.top + d.bottom) / 2f,
        d.right - d.left,
        d.bottom - d.top,
    )

    private fun compatible(track: Track, d: Detection, g: Geometry): Boolean {
        val distance = centerDistance(track.centerX, track.centerY, g.centerX, g.centerY)
        if (track.classId == d.classId) {
            val sameScale = scaleRatio(track.width, g.width) in MIN_SCALE_RATIO..MAX_SCALE_RATIO &&
                scaleRatio(track.height, g.height) in MIN_SCALE_RATIO..MAX_SCALE_RATIO
            return (distance <= SAME_CLASS_MAX_DISTANCE && intersectionOverUnion(track, d) >= SAME_CLASS_MIN_IOU) ||
                (distance <= SAME_CLASS_NEAR_DISTANCE && sameScale)
        }
        // Cross-class recovery is restricted to vehicle-like classes and requires strong overlap.
        return track.classId in VEHICLE_CLASSES && d.classId in VEHICLE_CLASSES &&
            distance <= CROSS_CLASS_MAX_DISTANCE && intersectionOverUnion(track, d) >= CROSS_CLASS_MIN_IOU
    }

    /** Reuse a recently ended alert only when same-class box geometry strongly agrees. */
    private fun compatibleRecent(track: Track, d: Detection, g: Geometry): Boolean {
        if (track.classId != d.classId) return false
        val distance = centerDistance(track.centerX, track.centerY, g.centerX, g.centerY)
        val overlap = intersectionOverUnion(track, d)
        val similarScale = scaleRatio(track.width, g.width) <= RECENT_MAX_SCALE_RATIO &&
            scaleRatio(track.height, g.height) <= RECENT_MAX_SCALE_RATIO
        return (distance <= RECENT_MAX_CENTER_DISTANCE && overlap >= RECENT_MIN_IOU) ||
            (distance <= RECENT_TIGHT_CENTER_DISTANCE && similarScale)
    }

    /** Coalesce only near-identical same-class boxes within one frame; strongest score wins. */
    private fun deduplicateFrameDetections(detections: List<Detection>): List<Detection> {
        val unique = mutableListOf<Detection>()
        for (detection in detections) {
            val detectionGeometry = geometry(detection)
            val duplicate = unique.any { kept ->
                if (kept.classId != detection.classId) return@any false
                val keptGeometry = geometry(kept)
                centerDistance(
                    keptGeometry.centerX,
                    keptGeometry.centerY,
                    detectionGeometry.centerX,
                    detectionGeometry.centerY,
                ) <=
                    FRAME_DUPLICATE_MAX_CENTER_DISTANCE &&
                    boxIntersectionOverUnion(kept, detection) >= FRAME_DUPLICATE_MIN_IOU
            }
            if (!duplicate) unique.add(detection)
        }
        return unique
    }

    private fun associationCost(track: Track, g: Geometry): Float =
        centerDistance(track.centerX, track.centerY, g.centerX, g.centerY)

    private fun intersectionOverUnion(track: Track, d: Detection): Float {
        val left = maxOf(track.centerX - track.width / 2f, d.left)
        val top = maxOf(track.centerY - track.height / 2f, d.top)
        val right = minOf(track.centerX + track.width / 2f, d.right)
        val bottom = minOf(track.centerY + track.height / 2f, d.bottom)
        val intersection = maxOf(0f, right - left) * maxOf(0f, bottom - top)
        val union = track.width * track.height + (d.right - d.left) * (d.bottom - d.top) - intersection
        return if (union > 0f) intersection / union else 0f
    }

    private fun boxIntersectionOverUnion(first: Detection, second: Detection): Float {
        val intersectionWidth = maxOf(0f, minOf(first.right, second.right) - maxOf(first.left, second.left))
        val intersectionHeight = maxOf(0f, minOf(first.bottom, second.bottom) - maxOf(first.top, second.top))
        val intersection = intersectionWidth * intersectionHeight
        val firstArea = (first.right - first.left) * (first.bottom - first.top)
        val secondArea = (second.right - second.left) * (second.bottom - second.top)
        val union = firstArea + secondArea - intersection
        return if (union > 0f) intersection / union else 0f
    }

    private fun directionOf(centerX: Float): Direction = when {
        centerX < CENTER_LEFT_BOUNDARY -> Direction.LEFT
        centerX > CENTER_RIGHT_BOUNDARY -> Direction.RIGHT
        else -> Direction.CENTER
    }

    private fun stableDirection(previous: Direction, centerX: Float): Direction = when (previous) {
        Direction.LEFT -> when {
            centerX > LEFT_EXIT_BOUNDARY -> if (centerX > RIGHT_ENTER_BOUNDARY) Direction.RIGHT else Direction.CENTER
            else -> Direction.LEFT
        }
        Direction.CENTER -> when {
            centerX < LEFT_ENTER_BOUNDARY -> Direction.LEFT
            centerX > RIGHT_ENTER_BOUNDARY -> Direction.RIGHT
            else -> Direction.CENTER
        }
        Direction.RIGHT -> when {
            centerX < RIGHT_EXIT_BOUNDARY -> if (centerX < LEFT_ENTER_BOUNDARY) Direction.LEFT else Direction.CENTER
            else -> Direction.RIGHT
        }
    }

    /** Image-area prominence is an internal escalation cue, not a spoken collision claim. */
    private fun isImportant(track: Track): Boolean =
        track.classId in IMPORTANT_CLASSES &&
            kotlin.math.abs(track.centerX - IMAGE_CENTER_X) <= IMPORTANT_CENTER_HALF_WIDTH &&
            track.width * track.height >= IMPORTANT_MIN_AREA

    private fun centerDistance(x1: Float, y1: Float, x2: Float, y2: Float): Float {
        val dx = x1 - x2
        val dy = y1 - y2
        return kotlin.math.sqrt(dx * dx + dy * dy)
    }

    private fun scaleRatio(a: Float, b: Float): Float = maxOf(a, b) / minOf(a, b)

    private fun Direction.label(): String = when (this) {
        Direction.LEFT -> "左侧"
        Direction.CENTER -> "前方"
        Direction.RIGHT -> "右侧"
    }

    private companion object {
        const val MIN_CONFIDENCE = 0.35f
        const val MAX_INPUT_DETECTIONS = 256
        const val MAX_DETECTIONS = 128
        const val CONFIRMATION_OBSERVATIONS = 2
        const val CONFIRMATION_MAX_GAP_MS = 500L
        const val LOSS_GRACE_MS = 2_500L
        const val REIDENTIFICATION_WINDOW_MS = 10_000L
        const val COOLDOWN_MS = 3_000L
        const val MAX_TRACKS = 128
        const val SAME_CLASS_MAX_DISTANCE = 0.18f
        const val SAME_CLASS_NEAR_DISTANCE = 0.12f
        const val SAME_CLASS_MIN_IOU = 0.10f
        const val MIN_SCALE_RATIO = 0.50f
        const val MAX_SCALE_RATIO = 2.00f
        const val CROSS_CLASS_MAX_DISTANCE = 0.08f
        const val CROSS_CLASS_MIN_IOU = 0.50f
        const val RECENT_MAX_CENTER_DISTANCE = 0.06f
        const val RECENT_TIGHT_CENTER_DISTANCE = 0.035f
        const val RECENT_MIN_IOU = 0.35f
        const val RECENT_MAX_SCALE_RATIO = 1.60f
        const val FRAME_DUPLICATE_MAX_CENTER_DISTANCE = 0.03f
        const val FRAME_DUPLICATE_MIN_IOU = 0.80f
        const val CENTER_LEFT_BOUNDARY = 0.38f
        const val CENTER_RIGHT_BOUNDARY = 0.62f
        const val LEFT_ENTER_BOUNDARY = 0.34f
        const val LEFT_EXIT_BOUNDARY = 0.42f
        const val RIGHT_EXIT_BOUNDARY = 0.58f
        const val RIGHT_ENTER_BOUNDARY = 0.66f
        const val IMAGE_CENTER_X = 0.5f
        const val IMPORTANT_CENTER_HALF_WIDTH = 0.18f
        const val IMPORTANT_MIN_AREA = 0.06f
        val VEHICLE_CLASSES = setOf(1, 2, 3, 5, 7)
        val IMPORTANT_CLASSES = setOf(1, 2, 3, 5, 7, 9, 11)
    }
}
