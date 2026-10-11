package org.sightguard.app

import org.sightguard.app.core.*

data class PresentationConfig(val cooldownMs: Long = 3000, val expiryMs: Long = 2000, val sameGeometryMemoryMs: Long = 10000, val directionMargin: Double = .05)
data class PresentedAlert(val alert: AlertEvent, val text: String, val important: Boolean, val merged: List<AlertEvent> = emptyList())

/** Mobile delivery policy is deliberately separate from the PC-compatible scheduler. */
class AlertPresenter(private val config: PresentationConfig = PresentationConfig()) {
    private data class Spoken(val classId: Int, val direction: Direction, val box: List<Double>, val sourceTimeMs: Double)
    private val directions = mutableMapOf<String, Direction>()
    private val spoken = mutableListOf<Spoken>()
    private var lastOrdinaryAt: Long? = null
    fun reset() { directions.clear(); spoken.clear(); lastOrdinaryAt = null }
    fun direction(id: String, center: Double): Direction {
        val previous = directions[id]
        val current = when (previous) {
            Direction.LEFT -> if (center <= .38 + config.directionMargin) Direction.LEFT else if (center >= .62 + config.directionMargin) Direction.RIGHT else Direction.CENTER
            Direction.RIGHT -> if (center >= .62 - config.directionMargin) Direction.RIGHT else if (center <= .38 - config.directionMargin) Direction.LEFT else Direction.CENTER
            Direction.CENTER -> if (center < .38 - config.directionMargin) Direction.LEFT else if (center > .62 + config.directionMargin) Direction.RIGHT else Direction.CENTER
            null -> if (center < .38) Direction.LEFT else if (center > .62) Direction.RIGHT else Direction.CENTER
        }
        directions[id] = current; return current
    }
    fun choose(result: StepResult, nowMs: Long): PresentedAlert? {
        return result.alerts.firstNotNullOfOrNull { chooseCandidate(it, result, nowMs) }
    }
    private fun chooseCandidate(selected: AlertEvent, result: StepResult, nowMs: Long): PresentedAlert? {
        val live = result.frame.events.filter { it.observedNow }.associateBy { it.eventId }
        directions.keys.retainAll(result.frame.events.map { it.eventId }.toSet())
        spoken.removeAll { result.frame.timestampMs - it.sourceTimeMs > config.sameGeometryMemoryMs }
        if (selected.alertType == "perception_limited") return PresentedAlert(selected, "感知暂时受限", true)
        val event = live[selected.eventId] ?: return null
        val box = event.assessment?.let { result.frame.observations.firstOrNull { o -> o.eventId == event.eventId }?.features?.spatial?.bboxNormalized } ?: return null
        val dir = direction(event.eventId, (box[0] + box[2]) / 2)
        val prefix = when (dir) { Direction.LEFT -> "左侧"; Direction.CENTER -> "前方"; Direction.RIGHT -> "右侧" }
        val upgrade = selected.alertType == "escalation"
        if (!upgrade && spoken.any { it.classId == event.classId && it.direction == dir && bboxIou(it.box, box) >= .55 }) return null
        val important = upgrade || (event.classId in setOf(1, 2, 3, 5, 7) && dir == Direction.CENTER && (event.assessment?.score ?: 0.0) >= .85)
        if (!important && lastOrdinaryAt?.let { nowMs - it < config.cooldownMs } == true) return null
        val peers = result.frame.events.filter { it.observedNow && it.state == EventState.ACTIVE && it.classId == event.classId && it.assessment!!.level.rank >= RiskLevel.MEDIUM.rank }
            .filter { other -> val geo = result.frame.observations.firstOrNull { it.eventId == other.eventId }?.features?.spatial ?: return@filter false; direction(other.eventId, geo.centerXY[0]) == dir }
        // Merge only currently eligible first notifications, never an unconfirmed box.
        val eligiblePeers = peers.filter { other -> result.alerts.any { it.eventId == other.eventId && it.alertType == "first" } }
        val text = when {
            upgrade -> "${prefix}目标风险升高，请注意"
            event.classId == 0 && eligiblePeers.size > 1 -> "${prefix}有多名行人"
            else -> "${prefix}有${Labels.coco80.getOrElse(event.classId) { "目标" }}"
        }
        return PresentedAlert(selected, text, important, if (event.classId == 0 && eligiblePeers.size > 1) result.alerts.filter { it.eventId != selected.eventId && it.eventId in eligiblePeers.map(EventSnapshot::eventId) } else emptyList())
    }
    fun accepted(presentation: PresentedAlert, result: StepResult, nowMs: Long) {
        if (!presentation.important) lastOrdinaryAt = nowMs
        val ids = (presentation.merged + presentation.alert).map { it.eventId }.toSet()
        result.frame.observations.filter { it.eventId in ids }.forEach { o ->
            val geo = o.features?.spatial ?: return@forEach
            spoken += Spoken(o.classId, direction(o.eventId!!, geo.centerXY[0]), geo.bboxNormalized, result.frame.timestampMs)
        }
    }
    fun deadline(nowMs: Long) = nowMs + config.expiryMs
}
