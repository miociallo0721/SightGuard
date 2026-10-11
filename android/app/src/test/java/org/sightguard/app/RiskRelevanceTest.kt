package org.sightguard.app

import org.junit.Assert.*
import org.junit.Test
import org.sightguard.app.core.*
import org.json.JSONObject

class RiskRelevanceTest {
    private fun detection(id: Int, track: Long? = 1L, box: List<Double> = listOf(200.0, 100.0, 800.0, 950.0)) = PDetection(track, id, Labels.coco80[id], .9, box)
    private fun frame(index: Int, time: Double, vararg detections: PDetection) = PFrame(index, time, 1000, 1000, detections.toList(), timestampSource = "video_pts")

    @Test fun contextualDetectionsRemainRawButDoNotCreateRisk() {
        val filter = RiskRelevance(); val engine = AssessmentEngine()
        for (index in 0..4) {
            val raw = frame(index, index * 250.0, *listOf(73, 64, 66, 63, 43).map { detection(it, it.toLong()) }.toTypedArray())
            val filtered = filter.apply(raw)
            assertEquals(5, raw.detections.size); assertTrue(filtered.frame.detections.isEmpty())
            assertTrue(filtered.decisions.all { it.category == RelevanceCategory.CONTEXTUAL && !it.eligible })
            assertTrue(engine.step(filtered.frame).frame.events.isEmpty())
            if (index == 0) assertTrue(filtered.decisions.single { it.classId == 43 }.classUncertain)
        }
        // Stable knife remains a raw knife label; this is not a pen classifier.
        val lowConfidenceKnife = detection(43).copy(confidence = .5)
        assertTrue(filter.apply(frame(5, 1250.0, lowConfidenceKnife)).decisions.single().classUncertain)
    }

    @Test fun realCocoConditionalIdsRequirePositionAndSourceTimePersistence() {
        assertEquals("椅子", Labels.coco80[56]); assertEquals("餐桌", Labels.coco80[60]); assertEquals("行李箱", Labels.coco80[28])
        for (id in listOf(56, 60, 28)) {
            val filter = RiskRelevance()
            assertFalse(filter.apply(frame(0, 0.0, detection(id))).decisions.single().eligible)
            assertFalse(filter.apply(frame(1, 250.0, detection(id))).decisions.single().eligible)
            assertTrue(filter.apply(frame(2, 500.0, detection(id))).decisions.single().eligible)
            val outside = detection(id, box = listOf(0.0, 0.0, 50.0, 50.0))
            assertFalse(filter.apply(frame(3, 750.0, outside)).decisions.single().eligible)
            assertFalse(filter.apply(frame(4, 1000.0, detection(id))).decisions.single().eligible)
        }
    }

    @Test fun lossesClassChangesMissingIdentityAndResetDoNotReuseConfirmation() {
        val filter = RiskRelevance()
        for (i in 0..2) filter.apply(frame(i, i * 250.0, detection(56)))
        filter.apply(frame(3, 600.0))
        assertFalse(filter.apply(frame(4, 700.0, detection(56))).decisions.single().eligible)
        assertFalse(filter.apply(frame(5, 800.0, detection(60))).decisions.single().eligible)
        assertFalse(filter.apply(frame(6, 1400.0, detection(60))).decisions.single().eligible)
        assertFalse(filter.apply(frame(7, 1700.0, detection(60, null))).decisions.single().eligible)
        filter.reset()
        assertFalse(filter.apply(frame(0, 0.0, detection(56))).decisions.single().eligible)
        assertThrows(IllegalArgumentException::class.java) { filter.apply(frame(1, 0.0).copy(timestampMs = 0.0)).let { filter.apply(frame(2, 1.0)); filter.apply(frame(3, .5)) } }
    }

    @Test fun primaryGeometryProducesExistingLowMediumHighAndEmptyMeansNoEvent() {
        fun level(box: List<Double>, id: Int): RiskLevel {
            val filter = RiskRelevance(); val engine = AssessmentEngine()
            engine.step(filter.apply(frame(0, 0.0, detection(id, box = box))).frame)
            return engine.step(filter.apply(frame(1, 300.0, detection(id, box = box))).frame).frame.events.single().assessment!!.level
        }
        assertEquals(RiskLevel.LOW, level(listOf(0.0, 0.0, 50.0, 50.0), 0))
        assertEquals(RiskLevel.MEDIUM, level(listOf(400.0, 200.0, 500.0, 400.0), 0))
        assertEquals(RiskLevel.HIGH, level(listOf(200.0, 100.0, 800.0, 950.0), 0))
        assertEquals(RiskLevel.HIGH, level(listOf(200.0, 100.0, 800.0, 950.0), 1))
        val engine = AssessmentEngine(); val filter = RiskRelevance()
        assertTrue(engine.step(filter.apply(frame(0, 0.0)).frame).frame.events.isEmpty())
    }

    @Test fun frozenRealCameraTraceRetainsPrimaryRecordsAndCoreOutput() {
        val lines = javaClass.getResourceAsStream("/camera_detection_trace.jsonl")!!.bufferedReader().use { it.readLines() }
        val tracker = LightTracker(); val relevance = RiskRelevance()
        val baseline = AssessmentEngine(); val filtered = AssessmentEngine()
        var records = 0
        lines.drop(1).filter { it.isNotBlank() }.forEachIndexed { index, line ->
            val json = JSONObject(line); val time = json.getLong("timestamp_ms")
            val rows = json.getJSONArray("detections")
            val raw = (0 until rows.length()).map { i -> val row = rows.getJSONArray(i)
                Detection(row.getInt(0), row.getDouble(1).toFloat(), row.getDouble(2).toFloat(), row.getDouble(3).toFloat(), row.getDouble(4).toFloat(), row.getDouble(5).toFloat())
            }
            val width = json.getInt("width"); val height = json.getInt("height")
            val detections = tracker.update(raw, time).map { tracked -> val d = tracked.detection
                PDetection(tracked.trackId, d.classId, Labels.coco80[d.classId], d.confidence.toDouble(), listOf(d.left * width.toDouble(), d.top * height.toDouble(), d.right * width.toDouble(), d.bottom * height.toDouble()))
            }
            val input = PFrame(index, time.toDouble(), width, height, detections, timestampSource = "monotonic")
            val result = relevance.apply(input)
            assertEquals(input.detections, result.frame.detections)
            assertEquals(baseline.step(input), filtered.step(result.frame))
            records += detections.size
        }
        assertTrue(records > 0)
    }

    @Test fun existingScoreBoundariesAndHysteresisRemainRealLevels() {
        // Reliable zero motion and zero appearance: raw=.45*focus+.20*motion.
        fun evidence(score: Double): RiskFeatures {
            val spatial = SpatialFeatures(listOf(.4, .1, .6, .3), listOf(.5, .2), .2, .2, 0.0, listOf(.5, .3), 1.0, false)
            val quality = PerceptionQuality(.9, 1.0, 4, 4, false, false, true, "video_pts", .9, emptyList())
            // Vary only valid image evidence; no score threshold or core configuration changes.
            val adjusted = if (score <= .45) spatial.copy(focusRelevance = score / .45)
                else spatial.copy(area = .2, bottomCenterXY = listOf(.5, .95), focusRelevance = (score - .35) / .45)
            return RiskFeatures(adjusted, 1000.0, 100.0, 0.0, listOf(0.0, 0.0), 0.0, 1.0, quality)
        }
        for ((score, expected) in listOf(.349 to RiskLevel.LOW, .35 to RiskLevel.MEDIUM, .649 to RiskLevel.MEDIUM, .65 to RiskLevel.HIGH)) {
            assertEquals(expected, RiskScorer(ScoringConfig()).assess("fresh", evidence(score), 0.0).level)
        }
        val scorer = RiskScorer(ScoringConfig())
        assertEquals(RiskLevel.MEDIUM, scorer.assess("stable", evidence(.60), 0.0).level)
        assertEquals(RiskLevel.MEDIUM, scorer.assess("stable", evidence(.66), 100.0).level)
        assertEquals(RiskLevel.HIGH, scorer.assess("stable", evidence(.68), 200.0).level)
        assertEquals(RiskLevel.HIGH, scorer.assess("stable", evidence(.64), 300.0).level)
        assertEquals(RiskLevel.HIGH, scorer.assess("stable", evidence(.59), 400.0).level)
        assertEquals(RiskLevel.HIGH, scorer.assess("stable", evidence(.59), 799.0).level)
        assertEquals(RiskLevel.MEDIUM, scorer.assess("stable", evidence(.59), 800.0).level)
    }

    @Test fun filteringDoesNotHideLimitedOrUnavailablePerception() {
        val filter = RiskRelevance(); val engine = AssessmentEngine()
        val invalid = detection(73, box = listOf(1200.0, 100.0, 1400.0, 300.0))
        val limited = engine.step(filter.apply(frame(0, 0.0, invalid)).frame)
        assertEquals("limited", limited.frame.perception.status)
        assertTrue(limited.frame.events.isEmpty())
        val failed = frame(1, 600.0).copy(status = "processing_error", error = "detector_failed")
        val unavailable = engine.step(filter.apply(failed).frame)
        assertEquals("unavailable", unavailable.frame.perception.status)
        assertTrue(unavailable.frame.events.isEmpty())
    }

    @Test fun duplicateTimestampsCannotManufactureConditionalConfirmation() {
        val filter = RiskRelevance()
        for (i in 0..10) assertFalse(filter.apply(frame(i, 0.0, detection(56))).decisions.single().eligible)
        assertFalse(filter.apply(frame(11, 500.0, detection(56))).decisions.single().eligible)
        assertTrue(filter.apply(frame(12, 750.0, detection(56))).decisions.single().eligible)
    }

    @Test fun initialMissingMotionCanProduceHighThenExistingHoldDelaysDowngrade() {
        val engine = AssessmentEngine(); val filter = RiskRelevance()
        val objectBox = detection(0, box = listOf(400.0, 200.0, 500.0, 600.0))
        fun step(index: Int, time: Double) = engine.step(filter.apply(frame(index, time, objectBox)).frame).frame.events.single().assessment!!
        val initial = step(0, 0.0)
        assertFalse(initial.quality.motionReliable)
        val scoring = engine.config.scoring
        val area = .1 * .4
        val appearance = scoring.appearanceAreaWeight * (area / scoring.areaFullScore) +
            (1.0 - scoring.appearanceAreaWeight) * ((.6 - scoring.bottomStart) / (scoring.bottomFullScore - scoring.bottomStart))
        val staticScore = scoring.spatialWeight + scoring.appearanceWeight * appearance
        assertEquals(staticScore / (scoring.spatialWeight + scoring.appearanceWeight), initial.score, 1e-9)
        assertEquals(RiskLevel.HIGH, initial.level)
        val confirmed = step(1, 300.0)
        assertTrue(confirmed.quality.motionReliable)
        assertEquals(staticScore, confirmed.score, 1e-9)
        assertEquals(RiskLevel.HIGH, confirmed.level)
        assertTrue("risk_level_held_by_hysteresis" in confirmed.reasons)
        assertEquals(RiskLevel.HIGH, step(2, 690.0).level)
        assertEquals(RiskLevel.MEDIUM, step(3, 700.0).level)
    }

    @Test fun configurationRejectsWrongCocoIdsAndOverlappingPolicies() {
        assertThrows(IllegalArgumentException::class.java) { RelevanceConfig(primaryClasses = setOf(80)) }
        assertThrows(IllegalArgumentException::class.java) { RelevanceConfig(uncertainClasses = setOf(-1)) }
        assertThrows(IllegalArgumentException::class.java) { RelevanceConfig(primaryClasses = setOf(56)) }
        assertThrows(IllegalArgumentException::class.java) { RelevanceConfig(conditionalMinArea = Double.NaN) }
        assertThrows(IllegalArgumentException::class.java) { RelevanceConfig(maxGapMs = 0.0) }
        assertThrows(IllegalArgumentException::class.java) { RelevanceConfig(memoryMs = 100.0) }
        val config = RelevanceConfig()
        assertTrue(listOf(73, 64, 66, 63, 43).none { it in config.primaryClasses || it in config.conditionalClasses })
    }
}
