package org.sightguard.app

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class RiskPolicyTest {
    @Test
    fun reportsImageSideAsDirectionProxy() {
        val policy = RiskPolicy()
        assertNull(policy.update(listOf(personAt(0.05f, 0.25f)), 0L))
        val alert = policy.update(listOf(personAt(0.05f, 0.25f)), 100L)
        assertEquals("左侧有行人", alert?.text)
        assertFalse(alert?.important ?: true)
    }

    @Test
    fun confirmsBeforeFirstAlertAndDoesNotRepeatWhileEventPersists() {
        val policy = RiskPolicy()
        assertNull(policy.update(listOf(personAt()), 0L))
        val first = policy.update(listOf(personAt()), 100L)
        assertNotNull(first)
        assertNull(policy.update(listOf(personAt()), 200L))
        assertNull(policy.update(listOf(personAt()), 300L))
    }

    @Test
    fun alertSuppressionBeforeSpeechReadyKeepsConfirmedEventEligible() {
        val policy = RiskPolicy()
        assertNull(policy.update(listOf(personAt()), 0L, allowAlerts = false))
        assertNull(policy.update(listOf(personAt()), 100L, allowAlerts = false))

        val firstAllowed = policy.update(listOf(personAt()), 200L, allowAlerts = true)
        assertNotNull(firstAllowed)
        assertNull(policy.update(listOf(personAt()), 300L, allowAlerts = true))
    }

    @Test
    fun recentSameBoxReappearanceKeepsEventAndOldHistoryEventuallyExpires() {
        val policy = RiskPolicy()
        policy.update(listOf(personAt()), 0L)
        val first = policy.update(listOf(personAt()), 100L)!!

        assertNull(policy.update(emptyList(), 300L))
        assertNull(policy.update(listOf(personAt()), 700L)) // Reacquired inside 2.5 s grace window.
        assertNull(policy.update(emptyList(), 800L))

        // Beyond active grace but inside 10 s, strong same-box evidence suppresses a duplicate.
        assertNull(policy.update(listOf(personAt()), 3_501L))
        assertNull(policy.update(listOf(personAt()), 3_601L))

        // Once the geometry history expires, a later appearance is a new confirmed event.
        assertNull(policy.update(listOf(personAt()), 13_602L))
        val second = policy.update(listOf(personAt()), 13_702L)
        assertNotNull(second)
        assertTrue(second!!.eventId > first.eventId)
    }

    @Test
    fun intermittentSameBoxDetectionsDoNotRepeatEveryCooldownAcrossThirtySeconds() {
        val policy = RiskPolicy()
        val target = personAt()
        policy.update(listOf(target), 0L)
        val first = policy.update(listOf(target), 100L)!!

        var timestamp = 100L
        repeat(10) {
            // Each gap exceeds active-loss grace but remains within recent geometry history.
            timestamp += 2_601L
            assertNull(policy.update(listOf(target), timestamp))
        }

        assertTrue(timestamp >= 26_000L)
        assertNull(policy.update(listOf(target), timestamp + 100L))
        assertEquals(1L, first.eventId)
    }

    @Test
    fun realCameraTraceDuplicateBusBoxesDoNotCreateASecondImportantEvent() {
        val policy = RiskPolicy()
        val initialBus = detection(
            classId = 5,
            confidence = 0.8320848f,
            left = 0.09370591f,
            top = 0.2684352f,
            right = 0.8421393f,
            bottom = 0.6439378f,
        )
        assertNull(policy.update(listOf(initialBus), 0L))
        val nextBusFrame = detection(
            5,
            0.8363137f,
            0.09229632f,
            0.27003306f,
            0.840432f,
            0.6435419f,
        )
        val firstAlert = policy.update(listOf(nextBusFrame), 275L)!!
        assertTrue(firstAlert.important)

        // Actual duplicate class-5 boxes from camera_detection_trace.jsonl at 4976 ms.
        val duplicateBoxesAt4976 = listOf(
            detection(5, 0.6462464f, 0.09837901f, 0.2666425f, 0.8420896f, 0.6386712f),
            detection(5, 0.4843253f, 0.09750061f, 0.26769453f, 0.84049606f, 0.6366354f),
        )
        assertNull(policy.update(duplicateBoxesAt4976, 4_976L))

        // The following real frame at 5042 ms also has two near-identical bus boxes.
        val duplicateBoxesAt5042 = listOf(
            detection(5, 0.6110128f, 0.09785913f, 0.2666369f, 0.84222037f, 0.6393044f),
            detection(5, 0.54433566f, 0.09696286f, 0.26763594f, 0.8407029f, 0.6374188f),
        )
        assertNull(policy.update(duplicateBoxesAt5042, 5_042L))
        val duplicateBoxesAt10384 = listOf(
            detection(5, 0.59495467f, 0.095766954f, 0.2673694f, 0.8405353f, 0.63838065f),
            detection(5, 0.5170317f, 0.09444447f, 0.2679394f, 0.8389801f, 0.63624555f),
        )
        assertNull(policy.update(duplicateBoxesAt10384, 10_384L))
        assertEquals(1L, firstAlert.eventId)
    }

    @Test
    fun replaysAllFramesFromRealCameraTraceWithoutRepeatingStableEvents() {
        val (modelSha, frames) = readCameraTrace()
        assertEquals("e093d7e1805d63cc1b671c333ce0b48b5cd33f70f64ea16c9b5bd713e33242cf", modelSha)
        assertEquals(200, frames.size)

        val policy = RiskPolicy()
        val alerts = frames.mapNotNull { frame ->
            policy.update(frame.detections, frame.timestampMs, allowAlerts = true)
        }

        assertEquals("Expected one bus event after suppressing duplicate same-frame boxes", 1,
            alerts.count { it.text.contains("公交车") })
        // The source photo has a fourth person clipped at the left edge, but the trace has
        // sustained detections for only three people; do not claim recall for an absent box.
        assertEquals("Expected one alert for each of the three sustained person tracks", 3,
            alerts.count { it.text.contains("行人") })
        assertEquals("No event should repeatedly alert during the continuous trace", 4, alerts.size)
    }

    @Test
    fun recentHistoryDoesNotMergeASeparateSameClassTarget() {
        val policy = RiskPolicy()
        policy.update(listOf(personAt()), 0L)
        val first = policy.update(listOf(personAt()), 100L)!!
        val separate = personAt(0.05f, 0.21f)

        assertNull(policy.update(listOf(separate), 3_200L))
        val second = policy.update(listOf(separate), 3_300L)
        assertNotNull(second)
        assertTrue(second!!.eventId > first.eventId)
    }

    @Test
    fun centeredLargeVehiclePreemptsRecentOrdinaryAlert() {
        val policy = RiskPolicy()
        policy.update(listOf(personAt(0.02f, 0.20f)), 0L)
        val ordinary = policy.update(listOf(personAt(0.02f, 0.20f)), 100L)!!
        assertFalse(ordinary.important)

        val car = carAtCenter()
        assertNull(policy.update(listOf(personAt(0.02f, 0.20f), car), 200L))
        val urgent = policy.update(listOf(personAt(0.02f, 0.20f), car), 300L)
        assertNotNull(urgent)
        assertTrue(urgent!!.important)
        assertTrue(urgent.eventId != ordinary.eventId)
    }

    @Test
    fun smallPositionJitterAcrossDirectionBoundaryKeepsOneEvent() {
        val policy = RiskPolicy()
        assertNull(policy.update(listOf(personAt(0.29f, 0.45f)), 0L)) // center x = .37, left
        val first = policy.update(listOf(personAt(0.31f, 0.47f)), 100L)!! // center x = .39
        assertEquals("左侧有行人", first.text)

        assertNull(policy.update(listOf(personAt(0.33f, 0.49f)), 700L))
        assertNull(policy.update(listOf(personAt(0.33f, 0.49f)), 1_300L))
        assertNull(policy.update(listOf(personAt(0.33f, 0.49f)), 1_400L))
    }

    @Test
    fun oneEventCanUpgradeOnceWhenItsImageRegionBecomesSignificant() {
        val policy = RiskPolicy()
        val smallCar = carAtCenter(0.45f, 0.40f, 0.55f, 0.60f)
        policy.update(listOf(smallCar), 0L)
        val first = policy.update(listOf(smallCar), 100L)!!
        assertFalse(first.important)

        val largeCar = carAtCenter()
        val upgrade = policy.update(listOf(largeCar), 200L)
        assertNotNull(upgrade)
        assertTrue(upgrade!!.important)
        assertEquals(first.eventId, upgrade.eventId)
        assertTrue(upgrade.text.contains("请注意"))
        assertNull(policy.update(listOf(largeCar), 1_400L))
    }

    @Test
    fun confidenceAboveOneIsRejected() {
        val policy = RiskPolicy()
        val invalid = personAt().copy(confidence = 1.01f)
        assertNull(policy.update(listOf(invalid), 0L))
        assertNull(policy.update(listOf(invalid), 100L))
    }

    @Test
    fun expireAndResetClearSessionState() {
        val policy = RiskPolicy()
        policy.update(listOf(personAt()), 0L)
        val first = policy.update(listOf(personAt()), 100L)!!

        policy.expire(10_101L)
        assertNull(policy.update(emptyList(), 10_101L))
        policy.reset()
        policy.update(listOf(personAt()), 0L)
        val afterRestart = policy.update(listOf(personAt()), 100L)!!
        assertEquals(1L, afterRestart.eventId)
        assertEquals(1L, first.eventId)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsBackwardsTime() {
        val policy = RiskPolicy()
        policy.update(emptyList(), 10L)
        policy.update(emptyList(), 9L)
    }

    private fun personAt(left: Float = 0.42f, right: Float = 0.58f) = Detection(
        classId = 0,
        confidence = 0.90f,
        left = left,
        top = 0.45f,
        right = right,
        bottom = 0.85f,
    )

    private fun carAtCenter(
        left: Float = 0.35f,
        top: Float = 0.25f,
        right: Float = 0.65f,
        bottom: Float = 0.75f,
    ) = Detection(
        classId = 2,
        confidence = 0.95f,
        left = left,
        top = top,
        right = right,
        bottom = bottom,
    )

    private fun detection(
        classId: Int,
        confidence: Float,
        left: Float,
        top: Float,
        right: Float,
        bottom: Float,
    ) = Detection(classId, confidence, left, top, right, bottom)

    private data class TraceFrame(val timestampMs: Long, val detections: List<Detection>)

    private fun readCameraTrace(): Pair<String, List<TraceFrame>> {
        val resource = javaClass.getResourceAsStream("/camera_detection_trace.jsonl")
            ?: error("Missing camera_detection_trace.jsonl test resource")
        val lines = resource.bufferedReader().use { it.readLines() }
        val header = lines.first()
        val sha = Regex("\"model_sha256\":\"([0-9a-f]{64})\"")
            .find(header)?.groupValues?.get(1)
            ?: error("Trace header is missing model_sha256")
        val timestampPattern = Regex("\"timestamp_ms\":(\\d+)")
        val detectionPattern = Regex(
            "\\[(-?\\d+),([-+0-9.eE]+),([-+0-9.eE]+),([-+0-9.eE]+),([-+0-9.eE]+),([-+0-9.eE]+)\\]",
        )

        val frames = lines.drop(1).map { line ->
            val timestamp = timestampPattern.find(line)?.groupValues?.get(1)?.toLong()
                ?: error("Trace frame is missing timestamp_ms")
            val detectionSection = line.substringAfter("\"detections\":[").substringBeforeLast(']')
            val detections = detectionPattern.findAll(detectionSection).map { match ->
                Detection(
                    classId = match.groupValues[1].toInt(),
                    confidence = match.groupValues[2].toFloat(),
                    left = match.groupValues[3].toFloat(),
                    top = match.groupValues[4].toFloat(),
                    right = match.groupValues[5].toFloat(),
                    bottom = match.groupValues[6].toFloat(),
                )
            }.toList()
            TraceFrame(timestamp, detections)
        }
        return sha to frames
    }
}
