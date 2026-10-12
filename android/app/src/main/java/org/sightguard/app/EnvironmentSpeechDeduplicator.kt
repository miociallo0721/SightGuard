package org.sightguard.app

import kotlin.math.max
import kotlin.math.min

/** Cross-source identity is deliberately short-lived and region-specific. Risk escalation is never suppressed. */
class EnvironmentSpeechDeduplicator(private val recoveryMs: Long = 4000) {
    private data class Heard(val generation: Long, val epoch: Long, val kind: EnvironmentKind,
        val bounds: List<Float>, val startedAtMs: Long)
    private val heard = mutableListOf<Heard>()
    fun reset() = heard.clear()

    /** Call only after actual YOLO speech onStart, never when a proposal is queued. */
    fun yoloStarted(generation: Long, epoch: Long, classId: Int, bounds: List<Double>, nowMs: Long, spokenText: String) {
        val kind = EnvironmentTaxonomy.yoloKind(classId) ?: return
        val className = Labels.detectionNames.getOrNull(classId) ?: return
        if (!spokenText.contains(className)) return // Generic fallback did not name this structure.
        val box = bounds.map { it.toFloat() }
        if (box.size != 4 || box.any { !it.isFinite() || it !in 0f..1f } || box[2] <= box[0] || box[3] <= box[1]) return
        heard.removeAll { nowMs - it.startedAtMs !in 0..recoveryMs }
        heard.add(Heard(generation, epoch, kind, box, nowMs))
    }

    fun alreadyHeard(generation: Long, epoch: Long, notice: EnvironmentNotice, nowMs: Long): Boolean {
        val support = notice.support ?: return false
        if (nowMs >= notice.validUntilMs || support.maskKeys.isEmpty()) return false
        heard.removeAll { nowMs - it.startedAtMs !in 0..recoveryMs }
        return heard.any { previous ->
            if (previous.generation != generation || previous.epoch != epoch || previous.kind != notice.kind) false
            else {
                val b = previous.bounds
                val intersection = max(0f, min(b[2], support.right) - max(b[0], support.left)) *
                    max(0f, min(b[3], support.bottom) - max(b[1], support.top))
                val union = (b[2]-b[0])*(b[3]-b[1]) + (support.right-support.left)*(support.bottom-support.top) - intersection
                val covered = support.maskKeys.count { key ->
                    val x = (key%64+.5f)/64; val y = (key/64+.5f)/64
                    x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]
                }.toFloat() / support.maskKeys.size
                union > 0f && intersection/union >= .55f && covered >= .6f
            }
        }
    }
}
