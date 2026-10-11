package org.sightguard.app.core

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

private fun clamp01(value: Double): Double = max(0.0, min(1.0, value))

private fun segmentDistance(point: Pair<Double, Double>, a: Pair<Double, Double>, b: Pair<Double, Double>): Double {
    val dx = b.first - a.first
    val dy = b.second - a.second
    val t = clamp01(((point.first - a.first) * dx + (point.second - a.second) * dy) / (dx * dx + dy * dy))
    return hypot(point.first - a.first - t * dx, point.second - a.second - t * dy)
}

/** Point-to-trapezoid relevance copied from the PC image-plane heuristic. */
fun focusRelevance(pointX: Double, pointY: Double, config: FeatureConfig): Double {
    val top = config.focusTopY
    val bottom = config.focusBottomY
    val halfTop = config.focusTopHalfWidth
    val halfBottom = config.focusBottomHalfWidth
    val centerX = config.focusCenterX
    val halfWidth = halfTop + (halfBottom - halfTop) * clamp01((pointY - top) / (bottom - top))
    if (pointY in top..bottom && abs(pointX - centerX) <= halfWidth) return 1.0

    val point = pointX to pointY
    val polygon = listOf(
        (centerX - halfTop) to top,
        (centerX + halfTop) to top,
        (centerX + halfBottom) to bottom,
        (centerX - halfBottom) to bottom,
    )
    val distance = polygon.indices.minOf { index ->
        segmentDistance(point, polygon[index], polygon[(index + 1) % polygon.size])
    }
    return clamp01(1.0 - distance / config.focusSoftMargin)
}

/** Coordinates are normalized and clipped to the image, matching sightguard.risk.features. */
fun spatialFeatures(detection: PDetection, width: Int, height: Int, config: FeatureConfig): SpatialFeatures {
    require(width > 0 && height > 0) { "Invalid frame dimensions" }
    val original = detection.bboxXyxy.mapIndexed { index, value -> value / if (index % 2 == 0) width else height }
    val box = original.map(::clamp01)
    val x1 = box[0]
    val y1 = box[1]
    val x2 = box[2]
    val y2 = box[3]
    require(x2 > x1 && y2 > y1) { "invalid_bbox: zero area or fully outside the image" }
    val centerX = (x1 + x2) / 2.0
    val centerY = (y1 + y2) / 2.0
    val normalizedWidth = x2 - x1
    val normalizedHeight = y2 - y1
    return SpatialFeatures(
        bboxNormalized = box,
        centerXY = listOf(centerX, centerY),
        width = normalizedWidth,
        height = normalizedHeight,
        area = normalizedWidth * normalizedHeight,
        bottomCenterXY = listOf(centerX, y2),
        focusRelevance = focusRelevance(centerX, y2, config),
        bboxClipped = box != original,
    )
}

class FeatureExtractor(val config: FeatureConfig) {
    private data class Opportunity(val timestampMs: Double, var observed: Boolean?)

    private data class History(
        val firstMs: Double,
        var lastMs: Double? = null,
        var spatial: SpatialFeatures? = null,
        var observations: Int = 0,
        val opportunities: MutableList<Opportunity> = mutableListOf(),
        var rates: List<Double>? = null,
        var motionBaselineMs: Double? = null,
        var motionBaselineSpatial: SpatialFeatures? = null,
    )

    private val histories = mutableMapOf<String, History>()

    fun spatialFeatures(detection: PDetection, width: Int, height: Int): SpatialFeatures =
        spatialFeatures(detection, width, height, config)

    fun beginFrame(frame: PFrame, complete: Boolean) {
        histories.values.forEach { history ->
            history.opportunities.add(Opportunity(frame.timestampMs, if (complete) false else null))
            while (history.opportunities.isNotEmpty() &&
                frame.timestampMs - history.opportunities.first().timestampMs > config.historyWindowMs
            ) {
                history.opportunities.removeAt(0)
            }
        }
    }

    fun forget(eventId: String) {
        histories.remove(eventId)
    }

    fun observe(
        eventId: String,
        detection: PDetection,
        frame: PFrame,
        spatial: SpatialFeatures,
    ): RiskFeatures {
        val history = histories.getOrPut(eventId) { History(frame.timestampMs) }
        if (history.opportunities.isEmpty() || history.opportunities.last().timestampMs != frame.timestampMs) {
            history.opportunities.add(Opportunity(frame.timestampMs, true))
        } else {
            history.opportunities.last().observed = true
        }

        val dt = history.lastMs?.let { frame.timestampMs - it }
        val previousOpportunity = history.opportunities.dropLast(1).lastOrNull()
        val interrupted = history.observations > 0 &&
            (previousOpportunity == null || previousOpportunity.observed != true || dt!! > config.motionMaxIntervalMs)

        val motionDt = history.motionBaselineMs?.let { frame.timestampMs - it }
        val resetBaseline = interrupted || motionDt == null || motionDt > config.motionMaxIntervalMs
        val elapsed = motionDt
        var rates: List<Double>? = null
        if (!resetBaseline && elapsed != null && elapsed >= config.motionMinIntervalMs &&
            elapsed <= config.motionMaxIntervalMs
        ) {
            val seconds = elapsed / 1000.0
            val previous = requireNotNull(history.motionBaselineSpatial)
            val previousX = previous.centerXY[0]
            val previousY = previous.centerXY[1]
            val centerX = spatial.centerXY[0]
            val centerY = spatial.centerXY[1]
            val rawRates = listOf(
                ln(sqrt(spatial.area / previous.area)) / seconds,
                (centerX - previousX) / seconds,
                (centerY - previousY) / seconds,
                (abs(previousX - config.focusCenterX) - abs(centerX - config.focusCenterX)) / seconds,
            )
            rates = if (config.motionSmoothingTauMs > 0.0 && history.rates != null) {
                val alpha = 1.0 - exp(-elapsed / config.motionSmoothingTauMs)
                history.rates!!.zip(rawRates) { old, new -> old + alpha * (new - old) }
            } else {
                rawRates
            }
        }

        if (resetBaseline || rates != null) {
            history.motionBaselineMs = frame.timestampMs
            history.motionBaselineSpatial = spatial
        }
        if (resetBaseline) history.rates = null else if (rates != null) history.rates = rates

        history.observations += 1
        val seen = history.opportunities.count { it.observed == true }
        val known = history.opportunities.count { it.observed != null }
        val continuity = if (known > 0) seen.toDouble() / known else 0.0
        val limitations = mutableListOf<String>()
        if (detection.trackId == null) limitations += "missing_track_id"
        if (interrupted) limitations += "observation_interrupted"
        if (rates == null) limitations += "motion_unavailable"
        if (spatial.bboxClipped) limitations += "bbox_clipped"
        if (frame.timestampSource == "fps_fallback") limitations += "timestamp_fps_fallback"

        var qualityScore = 0.5 * detection.confidence +
            0.3 * continuity +
            0.2 * min(1.0, seen.toDouble() / config.qualityObservationTarget)
        if (detection.trackId == null) qualityScore *= 0.85
        if (interrupted) qualityScore *= 0.75
        if (frame.timestampSource == "fps_fallback") qualityScore *= config.fallbackTimeQuality
        val quality = PerceptionQuality(
            detectionConfidence = detection.confidence,
            continuity = continuity,
            validObservations = history.observations,
            recentObservations = seen,
            trackIdMissing = detection.trackId == null,
            interrupted = interrupted,
            motionReliable = rates != null,
            timestampSource = frame.timestampSource,
            score = clamp01(qualityScore),
            limitations = limitations,
        )
        val result = RiskFeatures(
            spatial = spatial,
            observationDurationMs = frame.timestampMs - history.firstMs,
            observationIntervalMs = dt,
            scaleChangePerS = rates?.get(0),
            centerChangePerS = rates?.let { listOf(it[1], it[2]) },
            inwardChangePerS = rates?.get(3),
            detectionStability = continuity,
            quality = quality,
            motionIntervalMs = if (rates != null) motionDt else null,
        )
        history.lastMs = frame.timestampMs
        history.spatial = spatial
        return result
    }
}
