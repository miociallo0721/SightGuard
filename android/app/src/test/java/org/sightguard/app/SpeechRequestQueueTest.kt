package org.sightguard.app

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SpeechRequestQueueTest {
    @Test
    fun queueKeepsOneRequestAndImportantRequestCanReplaceNormal() {
        val queue = SpeechRequestQueue()
        val first = request("normal-1")
        val second = request("normal-2")
        val important = request("important", important = true)

        assertTrue(queue.offer(first).accepted)
        assertFalse(queue.offer(second).accepted)
        val replacement = queue.offer(important)

        assertTrue(replacement.accepted)
        assertEquals(first, replacement.replaced)
        assertEquals(important, queue.take())
        assertNull(queue.peek())
    }

    @Test
    fun importantQueuedRequestCannotBeReplacedByNormalRequest() {
        val queue = SpeechRequestQueue()
        val important = request("important", important = true)

        assertTrue(queue.offer(important).accepted)
        assertFalse(queue.offer(request("normal")).accepted)
        assertEquals(important, queue.peek())
    }

    @Test
    fun expiredWaitingRequestIsRemovedBeforeItBlocksNewWork() {
        val queue = SpeechRequestQueue()
        val expired = request("expired", validUntilElapsedMs = 100L)

        assertTrue(queue.offer(expired).accepted)
        assertEquals(expired, queue.discardExpired(nowElapsedMs = 100L))
        assertNull(queue.peek())
        assertTrue(queue.offer(request("retry")).accepted)
    }

    @Test
    fun importantOnlyPreemptsAnActiveNormalRequest() {
        val activeNormal = request("active")
        val activeImportant = request("active-important", important = true)

        assertFalse(SpeechRequestRules.preempts(request("normal-next"), activeNormal))
        assertTrue(SpeechRequestRules.preempts(request("important-next", important = true), activeNormal))
        assertFalse(SpeechRequestRules.preempts(request("important-next", important = true), activeImportant))
    }

    @Test
    fun expiryBoundaryIsExclusive() {
        val request = request("short-lived", validUntilElapsedMs = 100L)

        assertFalse(SpeechRequestRules.isExpired(request, nowElapsedMs = 99L))
        assertTrue(SpeechRequestRules.isExpired(request, nowElapsedMs = 100L))
        assertTrue(SpeechRequestRules.isExpired(request, nowElapsedMs = 101L))
    }

    @Test
    fun localPromptUsesOnlyAnExactClipAndReportsItsActualWords() {
        val exact = SpeechPromptCatalog.resolve("前方有汽车，请注意")
        val multiple = SpeechPromptCatalog.resolve("右侧有多名行人")
        val unmatched = SpeechPromptCatalog.resolve("前方有马桶")

        assertEquals("alert_center_car_important", exact.resourceName)
        assertEquals("前方有汽车，请注意", exact.spokenText)
        assertFalse(exact.usedGeneric)
        assertEquals("alert_right_multiple_pedestrians", multiple.resourceName)
        assertEquals("右侧有多名行人", multiple.spokenText)
        assertFalse(multiple.usedGeneric)
        assertEquals("alert_center_target", unmatched.resourceName)
        assertEquals("前方有目标", unmatched.spokenText)
        assertTrue(unmatched.usedGeneric)
    }

    @Test
    fun localCatalogCoversRiskEscalationAndCameraHealthPhrases() {
        val escalationCases = listOf(
            "左侧目标风险升高，请注意" to "alert_left_risk_elevated",
            "前方目标风险升高，请注意" to "alert_center_risk_elevated",
            "右侧目标风险升高，请注意" to "alert_right_risk_elevated",
        )
        val healthCases = listOf(
            "感知暂时受限" to "perception_limited",
            "感知已恢复" to "perception_recovered",
            "摄像头已停止" to "camera_stopped",
        )

        for ((text, resourceName) in escalationCases + healthCases) {
            val selection = SpeechPromptCatalog.resolve(text)
            assertEquals(resourceName, selection.resourceName)
            assertEquals(text, selection.spokenText)
            assertFalse(selection.usedGeneric)
        }
    }

    private fun request(
        eventId: String,
        important: Boolean = false,
        validUntilElapsedMs: Long = 1_000L,
    ) = SpeechRequest(
        text = eventId,
        important = important,
        eventId = eventId,
        validUntilElapsedMs = validUntilElapsedMs,
    )
}
