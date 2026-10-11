package org.sightguard.app.core

/** Serializable detection record corresponding to sightguard.data.Detection. */
data class PDetection(
    val trackId: Long?,
    val classId: Int,
    val className: String,
    val confidence: Double,
    val bboxXyxy: List<Double>,
) {
    init {
        require(trackId == null || trackId >= 0L) { "trackId must be non-negative or null" }
        require(classId >= 0 && className.isNotEmpty()) { "Invalid class" }
        require(confidence.isFinite() && confidence in 0.0..1.0) { "Invalid confidence" }
        require(bboxXyxy.size == 4 && bboxXyxy.all(Double::isFinite)) {
            "bboxXyxy must contain four finite coordinates"
        }
        require(bboxXyxy[2] >= bboxXyxy[0] && bboxXyxy[3] >= bboxXyxy[1]) {
            "Inverted bounding box"
        }
    }
}

/** One timestamped, replayable perception result; fields mirror sightguard.data.FrameRecord. */
data class PFrame(
    val frameIndex: Int,
    val timestampMs: Double,
    val frameWidth: Int,
    val frameHeight: Int,
    val detections: List<PDetection> = emptyList(),
    val status: String = "ok",
    val error: String? = null,
    val timestampSource: String = "monotonic",
    val schemaVersion: String = "1.0",
) {
    init {
        require(schemaVersion == PERCEPTION_SCHEMA_VERSION) { "Unsupported perception schemaVersion" }
        require(frameIndex >= 0) { "Invalid frameIndex" }
        require(timestampMs.isFinite() && timestampMs >= 0.0) { "Invalid timestampMs" }
        require(frameWidth > 0 && frameHeight > 0) { "Invalid frame dimensions" }
        require(timestampSource in TIMESTAMP_SOURCES) { "Invalid timestampSource" }
        require(status == "ok" || status == "processing_error") { "Invalid frame status" }
        require(status != "processing_error" || (!error.isNullOrEmpty() && detections.isEmpty())) {
            "Failed frames require an error and no detections"
        }
        require(status != "ok" || error == null) { "Successful frames cannot contain an error" }
    }

    companion object {
        const val PERCEPTION_SCHEMA_VERSION = "1.0"
        val TIMESTAMP_SOURCES = setOf("video_pts", "fps_fallback", "monotonic")
    }
}

data class SpatialFeatures(
    val bboxNormalized: List<Double>,
    val centerXY: List<Double>,
    val width: Double,
    val height: Double,
    val area: Double,
    val bottomCenterXY: List<Double>,
    val focusRelevance: Double,
    val bboxClipped: Boolean,
)

data class PerceptionQuality(
    val detectionConfidence: Double,
    val continuity: Double,
    val validObservations: Int,
    val recentObservations: Int,
    val trackIdMissing: Boolean,
    val interrupted: Boolean,
    val motionReliable: Boolean,
    val timestampSource: String,
    val score: Double,
    val limitations: List<String>,
)

data class RiskFeatures(
    val spatial: SpatialFeatures,
    val observationDurationMs: Double,
    val observationIntervalMs: Double?,
    val scaleChangePerS: Double?,
    val centerChangePerS: List<Double>?,
    val inwardChangePerS: Double?,
    val detectionStability: Double,
    val quality: PerceptionQuality,
    val motionIntervalMs: Double? = null,
)

enum class RiskLevel(val rank: Int) {
    LOW(0),
    MEDIUM(1),
    HIGH(2),
}

data class RiskAssessment(
    val spatialScore: Double,
    val appearanceScore: Double,
    val motionScore: Double?,
    val rawScore: Double,
    val score: Double,
    val level: RiskLevel,
    val quality: PerceptionQuality,
    val degraded: Boolean,
    val reasons: List<String>,
)
