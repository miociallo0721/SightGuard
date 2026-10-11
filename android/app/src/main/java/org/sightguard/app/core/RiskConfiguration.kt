package org.sightguard.app.core

private fun requireRange(name: String, value: Double, minimum: Double = 0.0, maximum: Double? = null) {
    require(value.isFinite() && value >= minimum && (maximum == null || value <= maximum)) {
        "Invalid $name: $value"
    }
}

data class FeatureConfig(
    val focusCenterX: Double = 0.5,
    val focusTopY: Double = 0.35,
    val focusBottomY: Double = 1.0,
    val focusTopHalfWidth: Double = 0.12,
    val focusBottomHalfWidth: Double = 0.45,
    val focusSoftMargin: Double = 0.15,
    val historyWindowMs: Double = 1500.0,
    val motionMinIntervalMs: Double = 10.0,
    val motionMaxIntervalMs: Double = 750.0,
    val motionSmoothingTauMs: Double = 150.0,
    val qualityObservationTarget: Int = 4,
    val fallbackTimeQuality: Double = 0.8,
) {
    init {
        listOf(
            "focusCenterX" to focusCenterX,
            "focusTopY" to focusTopY,
            "focusBottomY" to focusBottomY,
            "focusTopHalfWidth" to focusTopHalfWidth,
            "focusBottomHalfWidth" to focusBottomHalfWidth,
            "fallbackTimeQuality" to fallbackTimeQuality,
        ).forEach { (name, value) -> requireRange(name, value, 0.0, 1.0) }
        require(focusTopY < focusBottomY) { "Focus top must be above bottom" }
        require(focusTopHalfWidth > 0.0 && focusTopHalfWidth <= focusBottomHalfWidth) {
            "Focus widths must be positive and widen toward bottom"
        }
        require(focusBottomHalfWidth <= minOf(focusCenterX, 1.0 - focusCenterX)) {
            "Focus region must fit inside the image"
        }
        listOf(
            "focusSoftMargin" to focusSoftMargin,
            "historyWindowMs" to historyWindowMs,
            "motionMinIntervalMs" to motionMinIntervalMs,
            "motionMaxIntervalMs" to motionMaxIntervalMs,
        ).forEach { (name, value) -> requireRange(name, value, 1e-9) }
        requireRange("motionSmoothingTauMs", motionSmoothingTauMs)
        require(motionMinIntervalMs <= motionMaxIntervalMs) { "Motion interval range is inverted" }
        require(qualityObservationTarget >= 1) { "qualityObservationTarget must be a positive integer" }
    }
}

data class ScoringConfig(
    val spatialWeight: Double = 0.45,
    val appearanceWeight: Double = 0.35,
    val motionWeight: Double = 0.20,
    val areaFullScore: Double = 0.20,
    val bottomStart: Double = 0.30,
    val bottomFullScore: Double = 0.95,
    val appearanceAreaWeight: Double = 0.70,
    val scaleRateFullScore: Double = 0.50,
    val inwardRateFullScore: Double = 0.25,
    val motionScaleWeight: Double = 0.75,
    val mediumThreshold: Double = 0.35,
    val highThreshold: Double = 0.65,
    val smoothingTauMs: Double = 0.0,
    val upgradeMargin: Double = 0.02,
    val downgradeMargin: Double = 0.05,
    val downgradeHoldMs: Double = 400.0,
) {
    init {
        listOf(
            "spatialWeight" to spatialWeight,
            "appearanceWeight" to appearanceWeight,
            "motionWeight" to motionWeight,
            "appearanceAreaWeight" to appearanceAreaWeight,
            "motionScaleWeight" to motionScaleWeight,
            "mediumThreshold" to mediumThreshold,
            "highThreshold" to highThreshold,
            "upgradeMargin" to upgradeMargin,
            "downgradeMargin" to downgradeMargin,
            "bottomStart" to bottomStart,
            "bottomFullScore" to bottomFullScore,
        ).forEach { (name, value) -> requireRange(name, value, 0.0, 1.0) }
        require(kotlin.math.abs(spatialWeight + appearanceWeight + motionWeight - 1.0) <= 1e-9) {
            "Scoring weights must sum to 1"
        }
        require(spatialWeight + appearanceWeight > 0.0) {
            "Missing-motion scoring needs spatial/appearance weight"
        }
        require(0.0 < mediumThreshold && mediumThreshold < highThreshold && highThreshold < 1.0) {
            "Risk thresholds must be ordered within (0,1)"
        }
        require(bottomStart < bottomFullScore) { "Bottom normalization range is inverted" }
        require(highThreshold + upgradeMargin <= 1.0) { "Upgrade threshold exceeds 1" }
        require(downgradeMargin < mediumThreshold) { "Downgrade margin prevents low-risk recovery" }
        listOf(
            "areaFullScore" to areaFullScore,
            "scaleRateFullScore" to scaleRateFullScore,
            "inwardRateFullScore" to inwardRateFullScore,
        ).forEach { (name, value) -> requireRange(name, value, 1e-9) }
        requireRange("smoothingTauMs", smoothingTauMs)
        requireRange("downgradeHoldMs", downgradeHoldMs)
    }
}

data class AssociationConfig(
    val recoveryWindowMs: Double = 750.0,
    val trackMaxCenterDistance: Double = 0.35,
    val allowClassSwitchForTrackId: Boolean = false,
    val geometryMaxCenterDistance: Double = 0.15,
    val geometryMinIou: Double = 0.10,
    val geometryMinScore: Double = 0.35,
    val iouWeight: Double = 0.60,
    val ambiguityMargin: Double = 0.08,
    val confirmationMs: Double = 250.0,
    val confirmationObservations: Int = 2,
    val confirmationMaxGapMs: Double = 500.0,
) {
    init {
        listOf(
            "recoveryWindowMs" to recoveryWindowMs,
            "trackMaxCenterDistance" to trackMaxCenterDistance,
            "geometryMaxCenterDistance" to geometryMaxCenterDistance,
            "confirmationMaxGapMs" to confirmationMaxGapMs,
        ).forEach { (name, value) -> requireRange(name, value, 1e-9) }
        listOf(
            "geometryMinIou" to geometryMinIou,
            "geometryMinScore" to geometryMinScore,
            "iouWeight" to iouWeight,
            "ambiguityMargin" to ambiguityMargin,
        ).forEach { (name, value) -> requireRange(name, value, 0.0, 1.0) }
        requireRange("confirmationMs", confirmationMs)
        require(confirmationObservations >= 2) { "Ordinary events require at least two observations" }
    }
}

data class AlertConfig(
    val minimumLevel: String = "MEDIUM",
    val ordinaryIntervalMs: Double = 3000.0,
    val escalationIntervalMs: Double = 1000.0,
    val escalationMinScoreDelta: Double = 0.05,
    val repeatEnabled: Boolean = false,
    val maxAlertsPerFrame: Int = 1,
    val perceptionConfirmMs: Double = 500.0,
    val limitedQualityThreshold: Double = 0.30,
    val maxFrameGapMs: Double = 1000.0,
) {
    init {
        require(minimumLevel in setOf("LOW", "MEDIUM", "HIGH")) { "Invalid risk level" }
        listOf("ordinaryIntervalMs" to ordinaryIntervalMs, "escalationIntervalMs" to escalationIntervalMs,
            "perceptionConfirmMs" to perceptionConfirmMs).forEach { (name, value) -> requireRange(name, value) }
        requireRange("maxFrameGapMs", maxFrameGapMs, 1e-9)
        listOf("escalationMinScoreDelta" to escalationMinScoreDelta,
            "limitedQualityThreshold" to limitedQualityThreshold).forEach { (name, value) ->
            requireRange(name, value, 0.0, 1.0)
        }
        require(maxAlertsPerFrame >= 1) { "maxAlertsPerFrame must be a positive integer" }
    }
}

/** Default values exactly mirror configs/risk.yaml; mobile callers can copy only the sections they override. */
data class RiskConfig(
    val schemaVersion: String = "2.0",
    val features: FeatureConfig = FeatureConfig(),
    val scoring: ScoringConfig = ScoringConfig(),
    val association: AssociationConfig = AssociationConfig(),
    val alerts: AlertConfig = AlertConfig(),
) {
    init {
        require(schemaVersion == "2.0") { "Unsupported Phase 2 configuration schema" }
    }

    /** Build an immutable mobile-specific config while retaining PC values for omitted sections. */
    fun withMobileOverrides(
        features: FeatureConfig = this.features,
        scoring: ScoringConfig = this.scoring,
        association: AssociationConfig = this.association,
        alerts: AlertConfig = this.alerts,
    ): RiskConfig = copy(features = features, scoring = scoring, association = association, alerts = alerts)
}
