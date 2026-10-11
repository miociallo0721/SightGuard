package org.sightguard.app

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DirectionalAudioTest {
    @Test fun queuedSpeechRetainsTypedDirectionAndExactSpeechRequirement() {
        val queue = SpeechRequestQueue()
        val request = SpeechRequest("unrelated text", false, "event-right", 5000,
            direction = Direction.RIGHT, requireExactSpeech = true)
        assertTrue(queue.offer(request).accepted)
        assertEquals(Direction.RIGHT, queue.take()?.direction)
        queue.offer(request)
        assertEquals(true, queue.take()?.requireExactSpeech)
    }

    @Test fun leftAndRightAreMirroredAndCenterIsBalanced() {
        val left = DirectionalAudio.mix(Direction.LEFT)
        val right = DirectionalAudio.mix(Direction.RIGHT)
        assertTrue(left.leftGain > left.rightGain)
        assertTrue(right.rightGain > right.leftGain)
        assertEquals(left.leftGain, right.rightGain, 0f)
        assertEquals(left.rightGain, right.leftGain, 0f)
        assertEquals(-left.ttsPan, right.ttsPan, 0f)
        assertEquals(DirectionalAudioMix(1f, 1f, 0f), DirectionalAudio.mix(Direction.CENTER))
    }

    @Test fun systemAnnouncementsHaveNoSideBias() {
        assertEquals(DirectionalAudio.mix(Direction.CENTER), DirectionalAudio.mix(null))
    }

    @Test fun configurationControlsBothLocalAndTtsMix() {
        assertEquals(DirectionalAudioMix(1f, .4f, -.6f), DirectionalAudio.mix(Direction.LEFT, DirectionalAudioConfig(.4f)))
        assertEquals(DirectionalAudioMix(0f, 1f, 1f), DirectionalAudio.mix(Direction.RIGHT, DirectionalAudioConfig(0f)))
    }

    @Test fun invalidGainCannotSilentlyRemoveDirectionOrAmplify() {
        listOf(-.1f, 1f, 2f, Float.NaN, Float.POSITIVE_INFINITY).forEach { invalid ->
            var rejected = false
            try { DirectionalAudioConfig(invalid) } catch (_: IllegalArgumentException) { rejected = true }
            assertTrue("gain=$invalid", rejected)
        }
    }
}
