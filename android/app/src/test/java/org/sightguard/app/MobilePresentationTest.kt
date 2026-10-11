package org.sightguard.app

import org.junit.Assert.*
import org.junit.Test
import org.sightguard.app.core.*

class MobilePresentationTest {
    private val config = RiskConfig().let { it.copy(association = it.association.copy(recoveryWindowMs = 2500.0), alerts = it.alerts.copy(maxAlertsPerFrame = 8)) }
    private fun person(id: Long, x: Double = 250.0) = PDetection(id, 0, "person", .9, listOf(x, 200.0, x + 120, 700.0))
    @Test fun directionsUseHysteresisAndRestartClearsThem() {
        val p = AlertPresenter()
        assertEquals(Direction.LEFT, p.direction("a", .2)); assertEquals(Direction.LEFT, p.direction("a", .4))
        assertEquals(Direction.CENTER, p.direction("a", .5)); assertEquals(Direction.CENTER, p.direction("a", .62))
        assertEquals(Direction.RIGHT, p.direction("a", .8)); assertEquals(Direction.RIGHT, p.direction("a", .6))
        p.reset(); assertEquals(Direction.CENTER, p.direction("a", .5))
    }
    @Test fun continuedAndBrieflyOccludedEventStaysQuietAndLongAbsenceRealerts() {
        val engine = AssessmentEngine(config); val presenter = AlertPresenter(); var index = 0
        fun step(t: Long, detections: List<PDetection>) = engine.step(PFrame(index++, t.toDouble(), 1000, 1000, detections), false)
        assertNull(presenter.choose(step(0, listOf(person(1))), 0))
        val first = step(300, listOf(person(1))); val alert = presenter.choose(first, 300)!!
        presenter.accepted(alert, first, 300); engine.acceptAlert(alert.alert)
        for (t in 400L..2400L step 200) assertNull(presenter.choose(step(t, listOf(person(1))), t))
        step(2500, emptyList()); assertNull(presenter.choose(step(3000, listOf(person(2))), 3000))
        step(13000, emptyList()); step(14000, listOf(person(3)))
        assertNotNull(presenter.choose(step(14300, listOf(person(3))), 14300))
    }
    @Test fun rejectedOrMutedDeliveryRemainsRetryableAndCooldownArbitratesDistinctEvents() {
        val engine = AssessmentEngine(config); val p = AlertPresenter()
        engine.step(PFrame(0, 0.0, 1000, 1000, listOf(person(1), person(2, 700.0))), false)
        val result = engine.step(PFrame(1, 300.0, 1000, 1000, listOf(person(1), person(2, 700.0))), false)
        val initial = p.choose(result, 300)!!
        assertEquals(initial.alert.eventId, p.choose(result, 1000)!!.alert.eventId)
        p.accepted(initial, result, 1000); engine.acceptAlert(initial.alert)
        val next = engine.step(PFrame(2, 600.0, 1000, 1000, listOf(person(1), person(2, 700.0))), false)
        assertNull(p.choose(next, 2000))
        assertNotNull(p.choose(next, 4100))
        assertEquals(6100L, p.deadline(4100))
    }
    @Test fun emptyInputIsQuietAndNewSessionStartsFresh() {
        val first = AssessmentEngine(config)
        for (i in 0..10) assertTrue(first.step(PFrame(i, i * 100.0, 640, 480)).alerts.isEmpty())
        first.finish(); assertTrue(first.finish().isEmpty())
        val second = AssessmentEngine(config)
        assertEquals("E000001", second.step(PFrame(0, 0.0, 1000, 1000, listOf(person(1)))).frame.events.single().eventId)
    }
    @Test fun importantNewCentralVehicleBypassesOrdinaryCooldown() {
        val engine = AssessmentEngine(config); val presenter = AlertPresenter()
        engine.step(PFrame(0, 0.0, 1000, 1000, listOf(person(1))), false)
        val first = engine.step(PFrame(1, 300.0, 1000, 1000, listOf(person(1))), false)
        val ordinary = presenter.choose(first, 300)!!
        assertFalse(ordinary.important)
        presenter.accepted(ordinary, first, 300); engine.acceptAlert(ordinary.alert)
        val bus = PDetection(2, 5, "bus", .95, listOf(350.0, 450.0, 650.0, 850.0))
        engine.step(PFrame(2, 400.0, 1000, 1000, listOf(person(1), bus)), false)
        val largerBus = bus.copy(bboxXyxy = listOf(100.0, 50.0, 900.0, 1000.0))
        val confirmed = engine.step(PFrame(3, 700.0, 1000, 1000, listOf(person(1), largerBus)), false)
        val urgent = presenter.choose(confirmed, 700)!!
        assertTrue(urgent.important)
        assertEquals(5, confirmed.frame.events.single { it.eventId == urgent.alert.eventId }.classId)
    }
    @Test fun portraitLandscapeAndQuarterTurnBitmapUseExactFitCoordinates() {
        val portrait = fitImage(360, 640, 504f, 384f)
        assertEquals(144f, portrait.x(0f), .001f); assertEquals(360f, portrait.x(360f), .001f)
        assertEquals(0f, portrait.y(0f), .001f); assertEquals(384f, portrait.y(640f), .001f)
        val landscape = fitImage(640, 360, 504f, 498f)
        assertEquals(0f, landscape.x(0f), .001f); assertEquals(107.25f, landscape.y(0f), .001f)
        // Rotation is applied before mapping: the decoder's 360x640 output uses portrait fit.
        assertEquals(portrait, fitImage(360, 640, 504f, 384f))
    }
    @Test fun genericLocalClipPreservesCoarseDirectionWithoutInventingCategory() {
        val left = SpeechPromptCatalog.resolve("左侧有马桶")
        assertEquals("左侧有目标", left.spokenText); assertTrue(left.usedGeneric)
    }
}
