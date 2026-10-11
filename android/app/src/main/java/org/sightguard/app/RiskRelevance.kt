package org.sightguard.app

import org.sightguard.app.core.*

enum class RelevanceCategory { PRIMARY, CONDITIONAL, CONTEXTUAL }

/** Android image-plane policy, versioned separately from the unchanged PC risk core. */
data class RelevanceConfig(
    val schemaVersion: String = "android-relevance-1.0",
    val primaryClasses: Set<Int> = setOf(0, 1, 2, 3, 5, 7, 15, 16, 17, 18, 19, 20, 21, 22, 23),
    val conditionalClasses: Set<Int> = setOf(13, 24, 28, 36, 56, 57, 58, 59, 60, 61, 68, 69, 71, 72),
    val conditionalMinFocus: Double = .8,
    val conditionalMinArea: Double = .04,
    val conditionalMinBottom: Double = .65,
    val confirmationMs: Double = 500.0,
    val confirmationObservations: Int = 3,
    val maxGapMs: Double = 500.0,
    val memoryMs: Double = 2500.0,
    val uncertainClasses: Set<Int> = setOf(43, 76),
    val labelMinConfidence: Double = .65,
) {
    init {
        require(schemaVersion == "android-relevance-1.0")
        require(primaryClasses.intersect(conditionalClasses).isEmpty())
        require((primaryClasses + conditionalClasses + uncertainClasses).all { it in 0..79 }) { "Relevance classes must be valid COCO80 IDs" }
        require(listOf(conditionalMinFocus, conditionalMinArea, conditionalMinBottom, labelMinConfidence).all { it.isFinite() && it in 0.0..1.0 })
        require(confirmationMs.isFinite() && confirmationMs >= 0 && confirmationObservations >= 2)
        require(maxGapMs.isFinite() && maxGapMs > 0 && memoryMs.isFinite() && memoryMs >= maxGapMs)
    }
}

data class RelevanceDecision(
    val detectionIndex: Int,
    val trackId: Long?,
    val classId: Int,
    val category: RelevanceCategory,
    val eligible: Boolean,
    val classUncertain: Boolean,
    val reasons: List<String>,
)

data class RelevanceResult(val frame: PFrame, val decisions: List<RelevanceDecision>)

/** Filters only the engine input. Original detection records remain untouched for replay. */
class RiskRelevance(private val config: RelevanceConfig = RelevanceConfig(), private val features: FeatureConfig = FeatureConfig()) {
    private data class Evidence(val classId: Int, var firstMs: Double, var lastMs: Double, var count: Int)
    private val evidence = mutableMapOf<Long, Evidence>()
    private var lastTime: Double? = null
    fun reset() { evidence.clear(); lastTime = null }

    fun apply(frame: PFrame): RelevanceResult {
        require(lastTime == null || frame.timestampMs >= lastTime!!) { "Relevance timestamp must not decrease" }
        val time = frame.timestampMs
        evidence.entries.removeAll { time - it.value.lastMs > config.memoryMs }
        val decisions = frame.detections.mapIndexed { index, detection ->
            val category = when (detection.classId) {
                in config.primaryClasses -> RelevanceCategory.PRIMARY
                in config.conditionalClasses -> RelevanceCategory.CONDITIONAL
                else -> RelevanceCategory.CONTEXTUAL
            }
            val spatial = try { spatialFeatures(detection, frame.frameWidth, frame.frameHeight, features) } catch (_: IllegalArgumentException) { null }
            val position = spatial != null && spatial.focusRelevance >= config.conditionalMinFocus && spatial.area >= config.conditionalMinArea && spatial.bottomCenterXY[1] >= config.conditionalMinBottom
            val old = detection.trackId?.let(evidence::get)
            val continuous = old != null && old.classId == detection.classId && time - old.lastMs <= config.maxGapMs && (category != RelevanceCategory.CONDITIONAL || position)
            val state = if (continuous) old!! else Evidence(detection.classId, time, time, 0)
            if (time > state.lastMs || state.count == 0) state.count++
            state.lastMs = time
            if (!position && category == RelevanceCategory.CONDITIONAL) { state.firstMs = time; state.count = 0 }
            detection.trackId?.let { evidence[it] = state }
            val stable = detection.trackId != null && state.count >= config.confirmationObservations && time - state.firstMs >= config.confirmationMs
            val uncertain = detection.classId in config.uncertainClasses && (!stable || detection.confidence < config.labelMinConfidence)
            val eligible = spatial != null && when (category) {
                RelevanceCategory.PRIMARY -> true
                RelevanceCategory.CONDITIONAL -> position && stable
                RelevanceCategory.CONTEXTUAL -> false
            }
            RelevanceDecision(index, detection.trackId, detection.classId, category, eligible, uncertain, listOf(
                when { spatial == null -> "invalid_image_geometry"; category == RelevanceCategory.CONTEXTUAL -> "context_only_no_automatic_mobility_alert"; category == RelevanceCategory.PRIMARY -> "primary_mobility_object"; !position -> "conditional_image_position_not_met"; !stable -> "conditional_temporal_confirmation_pending"; else -> "conditional_image_position_and_persistence_confirmed" },
            ) + if (uncertain) listOf("class_uncertain_raw_label_retained") else emptyList())
        }
        // Missing observations break confirmation, while short-loss identity memory remains.
        val seen = frame.detections.mapNotNull { it.trackId }.toSet()
        evidence.filterKeys { it !in seen }.values.forEach { it.count = 0; it.firstMs = time }
        lastTime = time
        // Retain invalid geometry for the core's independent LIMITED-quality diagnostics;
        // the core rejects these observations rather than scoring them as risk.
        return RelevanceResult(frame.copy(detections = frame.detections.filterIndexed { index, _ ->
            decisions[index].eligible || "invalid_image_geometry" in decisions[index].reasons
        }), decisions)
    }
}
