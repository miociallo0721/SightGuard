package org.sightguard.app.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.assertThrows
import org.junit.Test

class RiskParityCoreTest {
    private fun detection(box: List<Double>, trackId: Long? = 7L): PDetection =
        PDetection(trackId, 0, "person", 0.9, box)

    private fun frame(index: Int, timestampMs: Double, detections: List<PDetection>): PFrame =
        PFrame(index, timestampMs, 640, 480, detections, timestampSource = "video_pts")

    private fun step(
        extractor: FeatureExtractor,
        index: Int,
        timestampMs: Double,
        item: PDetection,
        eventId: String = "E000001",
    ): RiskFeatures {
        val record = frame(index, timestampMs, listOf(item))
        extractor.beginFrame(record, complete = true)
        val geometry = extractor.spatialFeatures(item, record.frameWidth, record.frameHeight)
        return extractor.observe(eventId, item, record, geometry)
    }

    private fun syntheticFeatures(focus: Double, area: Double, bottomY: Double, scaleRate: Double, inwardRate: Double): RiskFeatures {
        val spatial = SpatialFeatures(
            bboxNormalized = listOf(0.4, 0.4, 0.6, 0.9),
            centerXY = listOf(0.5, 0.65),
            width = 0.2,
            height = 0.5,
            area = area,
            bottomCenterXY = listOf(0.5, bottomY),
            focusRelevance = focus,
            bboxClipped = false,
        )
        val quality = PerceptionQuality(0.9, 1.0, 3, 3, false, false, true, "video_pts", 0.9, emptyList())
        return RiskFeatures(spatial, 500.0, 250.0, scaleRate, listOf(0.0, 0.0), inwardRate, 1.0, quality)
    }

    @Test
    fun defaultsMatchPcRiskConfigurationAndMobileOverridesAreIsolated() {
        assertEquals(0.45, RiskConfig().scoring.spatialWeight, 0.0)
        assertEquals(1500.0, RiskConfig().features.historyWindowMs, 0.0)
        assertEquals(750.0, RiskConfig().association.recoveryWindowMs, 0.0)
        assertEquals("MEDIUM", RiskConfig().alerts.minimumLevel)

        val mobile = RiskConfig().withMobileOverrides(scoring = RiskConfig().scoring.copy(highThreshold = 0.6))
        assertEquals(0.6, mobile.scoring.highThreshold, 0.0)
        assertEquals(0.65, RiskConfig().scoring.highThreshold, 0.0)
    }

    @Test
    fun perceptionContractsRejectInvalidRecords() {
        assertThrows(IllegalArgumentException::class.java) {
            PDetection(-1L, 0, "person", 0.9, listOf(1.0, 2.0, 3.0, 4.0))
        }
        assertThrows(IllegalArgumentException::class.java) {
            PDetection(null, 0, "person", 0.9, listOf(1.0, 2.0, 3.0))
        }
        assertThrows(IllegalArgumentException::class.java) {
            PFrame(0, 0.0, 640, 480, status = "processing_error")
        }
        assertThrows(IllegalArgumentException::class.java) {
            PFrame(0, Double.NaN, 640, 480)
        }
    }

    @Test
    fun normalizedSpatialFeaturesMatchPcCoordinatesAndFocus() {
        val item = detection(listOf(280.0, 300.0, 360.0, 450.0))
        val spatial = spatialFeatures(item, 640, 480, FeatureConfig())

        assertEquals(listOf(0.4375, 0.625, 0.5625, 0.9375), spatial.bboxNormalized)
        assertEquals(0.5, spatial.centerXY[0], 1e-12)
        assertEquals(0.78125, spatial.centerXY[1], 1e-12)
        assertEquals(0.0390625, spatial.area, 1e-12)
        assertEquals(1.0, spatial.focusRelevance, 0.0)
        assertFalse(spatial.bboxClipped)
    }

    @Test
    fun motionUsesIrregularVideoTimestampIntervals() {
        val extractor = FeatureExtractor(FeatureConfig(motionSmoothingTauMs = 0.0))
        step(extractor, 0, 20.0, detection(listOf(300.0, 300.0, 320.0, 340.0)))
        val doubled = step(extractor, 1, 220.0, detection(listOf(290.0, 290.0, 330.0, 370.0)))
        val redoubled = step(extractor, 2, 570.0, detection(listOf(270.0, 250.0, 350.0, 410.0)))

        assertEquals(3.4657359, doubled.scaleChangePerS!!, 1e-6)
        assertEquals(1.9804209, redoubled.scaleChangePerS!!, 1e-6)
        assertTrue(doubled.quality.motionReliable)
        assertEquals(200.0, doubled.motionIntervalMs!!, 0.0)
        assertEquals(350.0, redoubled.motionIntervalMs!!, 0.0)
    }

    @Test
    fun completeMissInterruptsMotionHistoryAndForgetStartsFreshState() {
        val extractor = FeatureExtractor(FeatureConfig(motionSmoothingTauMs = 0.0))
        step(extractor, 0, 0.0, detection(listOf(280.0, 300.0, 360.0, 450.0)))
        extractor.beginFrame(frame(1, 200.0, emptyList()), complete = true)
        val afterLossFrame = frame(2, 400.0, listOf(detection(listOf(270.0, 290.0, 370.0, 455.0))))
        extractor.beginFrame(afterLossFrame, complete = true)
        val resumedDetection = afterLossFrame.detections.single()
        val interrupted = extractor.observe("E000001", resumedDetection, afterLossFrame,
            extractor.spatialFeatures(resumedDetection, 640, 480))

        assertTrue(interrupted.quality.interrupted)
        assertFalse(interrupted.quality.motionReliable)
        assertNull(interrupted.scaleChangePerS)
        extractor.forget("E000001")
        val fresh = step(extractor, 3, 600.0, detection(listOf(280.0, 300.0, 360.0, 450.0)))
        assertEquals(0.0, fresh.observationDurationMs, 0.0)
        assertEquals(1, fresh.quality.validObservations)
    }

    @Test
    fun missingMotionRenormalizesScoresAndDoesNotMixPerceptionQuality() {
        val extractor = FeatureExtractor(FeatureConfig())
        val features = step(extractor, 0, 0.0, detection(listOf(280.0, 300.0, 360.0, 450.0)))
        val scorer = RiskScorer(ScoringConfig())
        val assessment = scorer.assess("E000001", features, 0.0)
        val lowQuality = features.copy(quality = features.quality.copy(score = 0.01))
        val other = RiskScorer(ScoringConfig()).assess("E000002", lowQuality, 0.0)
        val expected = (
            ScoringConfig().spatialWeight * assessment.spatialScore +
                ScoringConfig().appearanceWeight * assessment.appearanceScore
            ) / (ScoringConfig().spatialWeight + ScoringConfig().appearanceWeight)

        assertNull(assessment.motionScore)
        assertTrue(assessment.degraded)
        assertEquals(expected, assessment.rawScore, 1e-12)
        assertEquals(assessment.rawScore, other.rawScore, 1e-12)
        assertEquals(0.01, other.quality.score, 0.0)
    }

    @Test
    fun fallbackTimestampAndMissingTrackIdStayVisibleInQuality() {
        val extractor = FeatureExtractor(FeatureConfig())
        val record = PFrame(
            frameIndex = 0,
            timestampMs = 0.0,
            frameWidth = 640,
            frameHeight = 480,
            detections = listOf(detection(listOf(280.0, 300.0, 360.0, 450.0), trackId = null)),
            timestampSource = "fps_fallback",
        )
        extractor.beginFrame(record, complete = true)
        val item = record.detections.single()
        val features = extractor.observe("E000001", item, record, extractor.spatialFeatures(item, 640, 480))

        assertTrue(features.quality.trackIdMissing)
        assertTrue(features.quality.limitations.contains("missing_track_id"))
        assertTrue(features.quality.limitations.contains("timestamp_fps_fallback"))
        assertFalse(features.quality.motionReliable)
    }

    @Test
    fun downgradeHysteresisUsesVideoTimeAndCanUpgradeImmediately() {
        val scorer = RiskScorer(ScoringConfig())
        val high = syntheticFeatures(1.0, 0.2, 0.95, 1.0, 1.0)
        val low = syntheticFeatures(0.0, 0.0, 0.30, -1.0, -1.0)

        assertEquals(RiskLevel.HIGH, scorer.assess("E000001", high, 0.0).level)
        assertEquals(RiskLevel.HIGH, scorer.assess("E000001", low, 100.0).level)
        assertEquals(RiskLevel.HIGH, scorer.assess("E000001", low, 499.0).level)
        assertEquals(RiskLevel.LOW, scorer.assess("E000001", low, 500.0).level)
        assertEquals(RiskLevel.HIGH, scorer.assess("E000001", high, 600.0).level)
    }

    @Test
    fun interruptionResetsPendingDowngradeAndEventStateIsScoped() {
        val scorer = RiskScorer(ScoringConfig())
        val high = syntheticFeatures(1.0, 0.2, 0.95, 1.0, 1.0)
        val low = syntheticFeatures(0.0, 0.0, 0.30, -1.0, -1.0)

        assertEquals(RiskLevel.HIGH, scorer.assess("event-a", high, 0.0).level)
        scorer.assess("event-a", low, 100.0)
        scorer.interrupt("event-a")
        assertEquals(RiskLevel.HIGH, scorer.assess("event-a", low, 500.0).level)
        assertEquals(RiskLevel.LOW, scorer.assess("event-a", low, 900.0).level)
        assertEquals(RiskLevel.LOW, scorer.assess("event-b", low, 500.0).level)
        scorer.forget("event-a")
        assertEquals(RiskLevel.LOW, scorer.assess("event-a", low, 901.0).level)
    }
}
