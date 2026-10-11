package org.sightguard.app

import org.junit.Assert.*
import org.junit.Test
import org.sightguard.app.core.*

/** Source-time core produces the levels; tests never relabel a score in the UI. */
class AlertPresenterLevelsTest {
    private val config = RiskConfig().let { it.copy(alerts = it.alerts.copy(maxAlertsPerFrame = 8)) }
    private fun medium(id: Long, x: Double = 400.0, classId: Int = 0) =
        PDetection(id, classId, Labels.coco80[classId], .9, listOf(x, 200.0, x + 100.0, 400.0))
    private fun high(id: Long, classId: Int = 0) =
        PDetection(id, classId, Labels.coco80[classId], .9, listOf(200.0, 100.0, 800.0, 950.0))
    private class Session(config: RiskConfig) {
        val engine = AssessmentEngine(config)
        val presenter = AlertPresenter()
        private var index = 0
        fun step(time: Long, vararg detections: PDetection) = engine.step(PFrame(index++, time.toDouble(), 1000, 1000, detections.toList()), false)
        fun accept(alert: PresentedAlert, result: StepResult, wallTime: Long) {
            presenter.accepted(alert, result, wallTime)
            (alert.merged + alert.alert).forEach(engine::acceptAlert)
        }
    }

    @Test fun lowIsSilentEvenWhenSchedulerIsConfiguredToExposeLowAlerts() {
        val session = Session(config.copy(alerts = config.alerts.copy(minimumLevel = "LOW")))
        val far = PDetection(1, 0, "person", .9, listOf(0.0, 0.0, 50.0, 50.0))
        session.step(0, far)
        val result = session.step(300, far)
        assertEquals(RiskLevel.LOW, result.frame.events.single().assessment!!.level)
        assertEquals(RiskLevel.LOW, result.alerts.single().riskLevel)
        assertNull(session.presenter.choose(result, 300))
    }

    @Test fun mediumFirstDeliveryUsesOrdinaryCooldownAndAcceptedEventDoesNotRepeat() {
        val session = Session(config)
        session.step(0, medium(1))
        val first = session.step(300, medium(1))
        val presentation = session.presenter.choose(first, 300)!!
        assertEquals(RiskLevel.MEDIUM, presentation.alert.riskLevel)
        assertFalse(presentation.important)
        // Muted/rejected playback has not committed anything and remains retryable.
        assertNotNull(session.presenter.choose(first, 400))
        session.accept(presentation, first, 400)
        assertNull(session.presenter.choose(session.step(600, medium(1)), 600))
        session.step(700, medium(1), medium(2, classId = 1))
        val other = session.step(1000, medium(1), medium(2, classId = 1))
        assertEquals(RiskLevel.MEDIUM, other.alerts.single().riskLevel)
        assertNull(session.presenter.choose(other, 1000))
        for (time in 1300L..3100L step 300L) {
            val ongoing = session.step(time, medium(1), medium(2, classId = 1))
            assertTrue(ongoing.frame.events.all { it.observedNow && it.state == EventState.ACTIVE })
            assertNull(session.presenter.choose(ongoing, time))
        }
        val current = session.step(3400, medium(1), medium(2, classId = 1))
        assertTrue(current.frame.events.all { it.observedNow && it.state == EventState.ACTIVE })
        val released = session.presenter.choose(current, 3400)!!
        assertFalse(released.important)
        session.accept(released, current, 3400)
        assertTrue(session.step(3700, medium(1), medium(2, classId = 1)).alerts.isEmpty())
    }

    @Test fun newHighPersonAndBicycleBypassOrdinaryCooldown() {
        for (classId in listOf(0, 1)) {
            val session = Session(config)
            session.step(0, medium(1, classId = 3))
            val ordinary = session.step(300, medium(1, classId = 3))
            val accepted = session.presenter.choose(ordinary, 300)!!
            assertFalse(accepted.important)
            session.accept(accepted, ordinary, 300)
            session.step(400, medium(1, classId = 3), high(2, classId))
            val urgentResult = session.step(700, medium(1, classId = 3), high(2, classId))
            val urgent = session.presenter.choose(urgentResult, 700)!!
            assertEquals(RiskLevel.HIGH, urgent.alert.riskLevel)
            assertTrue(urgent.important)
            assertEquals(classId, urgentResult.frame.events.single { it.eventId == urgent.alert.eventId }.classId)
            session.accept(urgent, urgentResult, 700)
            assertNull(session.presenter.choose(session.step(1000, medium(1, classId = 3), high(2, classId)), 1000))
        }
    }

    @Test fun confirmedMediumToHighEscalationBypassesSameGeometryMemoryAndOrdinaryCooldown() {
        val session = Session(config)
        session.step(0, medium(1))
        val result = session.step(300, medium(1))
        val accepted = session.presenter.choose(result, 300)!!
        session.accept(accepted, result, 300)
        val enlarged = medium(1).copy(bboxXyxy = listOf(250.0, 150.0, 650.0, 900.0))
        // Maintain reliable event identity through moderate geometry changes.
        session.step(700, enlarged)
        val escalation = session.step(1300, enlarged)
        val presentation = session.presenter.choose(escalation, 1300)!!
        assertEquals(accepted.alert.eventId, presentation.alert.eventId)
        assertEquals("escalation", presentation.alert.alertType)
        assertEquals(RiskLevel.HIGH, presentation.alert.riskLevel)
        assertEquals("前方目标风险升高，请注意", presentation.text)
        assertTrue(presentation.important)
        session.accept(presentation, escalation, 1300)
        assertTrue(session.step(1600, enlarged).alerts.isEmpty())
    }

    @Test fun aggregationCommitsOnlyConfirmedEligibleSameDirectionPeople() {
        val session = Session(config)
        val leftPerson = medium(1, 420.0)
        val rightPerson = medium(2, 540.0)
        session.step(0, leftPerson, rightPerson)
        val confirmed = session.step(300, leftPerson, rightPerson, medium(3, 480.0))
        val merged = session.presenter.choose(confirmed, 300)!!
        assertEquals("前方有多名行人", merged.text)
        assertEquals(1, merged.merged.size)
        val candidate = confirmed.frame.events.single { !it.confirmed }
        assertFalse(candidate.eventId in (merged.merged + merged.alert).map { it.eventId })
        session.accept(merged, confirmed, 300)
        val next = session.step(400, leftPerson, rightPerson)
        assertTrue(next.alerts.isEmpty())
        assertNull(session.presenter.choose(next, 400))
    }
}
