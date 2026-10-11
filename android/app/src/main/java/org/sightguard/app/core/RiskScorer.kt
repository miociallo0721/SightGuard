package org.sightguard.app.core

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

private fun clampScore(value: Double): Double = max(0.0, min(1.0, value))

class RiskScorer(val config: ScoringConfig) {
    private data class ScoreState(
        var score: Double,
        var level: RiskLevel,
        var lastMs: Double,
        var pendingLevel: RiskLevel? = null,
        var pendingSinceMs: Double? = null,
    )

    private val states = mutableMapOf<String, ScoreState>()

    fun forget(eventId: String) {
        states.remove(eventId)
    }

    fun interrupt(eventId: String) {
        states[eventId]?.let {
            it.pendingLevel = null
            it.pendingSinceMs = null
        }
    }

    private fun level(score: Double): RiskLevel = when {
        score >= config.highThreshold -> RiskLevel.HIGH
        score >= config.mediumThreshold -> RiskLevel.MEDIUM
        else -> RiskLevel.LOW
    }

    fun assess(eventId: String, features: RiskFeatures, timestampMs: Double): RiskAssessment {
        val c = config
        val spatial = features.spatial.focusRelevance
        val area = clampScore(features.spatial.area / c.areaFullScore)
        val bottom = clampScore(
            (features.spatial.bottomCenterXY[1] - c.bottomStart) / (c.bottomFullScore - c.bottomStart),
        )
        val appearance = c.appearanceAreaWeight * area + (1.0 - c.appearanceAreaWeight) * bottom
        val motion = if (features.quality.motionReliable) {
            c.motionScaleWeight * clampScore(max(0.0, requireNotNull(features.scaleChangePerS)) / c.scaleRateFullScore) +
                (1.0 - c.motionScaleWeight) * clampScore(
                    max(0.0, requireNotNull(features.inwardChangePerS)) / c.inwardRateFullScore,
                )
        } else {
            null
        }
        val numerator = c.spatialWeight * spatial + c.appearanceWeight * appearance
        val raw = clampScore(
            if (motion == null) numerator / (c.spatialWeight + c.appearanceWeight)
            else numerator + c.motionWeight * motion,
        )
        val previous = states[eventId]
        var score = raw
        if (previous != null && c.smoothingTauMs > 0.0) {
            val alpha = 1.0 - exp(-(timestampMs - previous.lastMs) / c.smoothingTauMs)
            score = previous.score + alpha * (raw - previous.score)
        }

        val reasons = mutableListOf<String>()
        if (spatial > 0.0) reasons += "image_focus_relevance"
        if (appearance > 0.0) reasons += "image_scale_and_bottom_position"
        if (motion == null) {
            reasons += "motion_unavailable_renormalized_spatial_appearance_weights"
        } else if (motion > 0.0) {
            reasons += "positive_scale_or_inward_image_change"
        } else {
            reasons += "no_positive_motion_evidence_static_risk_retained"
        }

        val target = level(score)
        val state = previous ?: ScoreState(score, target, timestampMs)
        if (previous != null) {
            if (features.quality.interrupted) interrupt(eventId)
            if (target.rank > state.level.rank) {
                val upgradeTarget = level(max(0.0, score - c.upgradeMargin))
                if (upgradeTarget.rank > state.level.rank) state.level = upgradeTarget
                state.pendingLevel = null
                state.pendingSinceMs = null
            } else if (target.rank < state.level.rank) {
                val lower = level(min(1.0, score + c.downgradeMargin))
                if (lower.rank < state.level.rank) {
                    if (state.pendingLevel != lower) {
                        state.pendingLevel = lower
                        state.pendingSinceMs = timestampMs
                    }
                    if (timestampMs - requireNotNull(state.pendingSinceMs) >= c.downgradeHoldMs) {
                        state.level = lower
                        state.pendingLevel = null
                        state.pendingSinceMs = null
                    }
                } else {
                    state.pendingLevel = null
                    state.pendingSinceMs = null
                }
            } else {
                state.pendingLevel = null
                state.pendingSinceMs = null
            }
            if (state.level != target) reasons += "risk_level_held_by_hysteresis"
        }
        state.score = score
        state.lastMs = timestampMs
        states[eventId] = state
        return RiskAssessment(
            spatialScore = spatial,
            appearanceScore = appearance,
            motionScore = motion,
            rawScore = raw,
            score = score,
            level = state.level,
            quality = features.quality,
            degraded = motion == null,
            reasons = reasons,
        )
    }
}
