package org.sightguard.app

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SpeechGateTest {
    @Test
    fun initializationFailureNeverAllowsSpeech() {
        val gate = SpeechGate()
        val session = gate.start()

        assertFalse(gate.canUseFallback)
        assertFalse(gate.initializationFinished(session, succeeded = false))
        assertFalse(gate.canSpeak)
        assertTrue(gate.isFallbackReady)
        assertTrue(gate.canUseFallback)
        gate.setMuted(true)
        assertFalse(gate.canUseFallback)
        gate.setMuted(false)
        assertTrue(gate.canUseFallback)
        assertFalse(gate.initializationFinished(session, succeeded = true))
    }

    @Test
    fun muteBlocksSpeechAndUnmuteRestoresReadySession() {
        val gate = readyGate()

        gate.setMuted(true)
        assertFalse(gate.canSpeak)

        gate.setMuted(false)
        assertTrue(gate.canSpeak)
    }

    @Test
    fun stopBlocksSpeechAndFinishingStopAllowsRestart() {
        val gate = readyGate()

        gate.beginStop()
        assertTrue(gate.isStopping)
        assertFalse(gate.canSpeak)
        assertFalse(gate.canUseFallback)
        gate.finishStop()
        assertFalse(gate.isFallbackReady)

        val nextSession = gate.start()
        assertTrue(nextSession > 1)
        assertTrue(gate.initializationFinished(nextSession, succeeded = true))
        assertTrue(gate.canSpeak)
    }

    @Test
    fun staleInitializationCannotReviveStoppedSession() {
        val gate = SpeechGate()
        val oldSession = gate.start()
        gate.beginStop()
        gate.finishStop()
        val newSession = gate.start()

        assertFalse(gate.initializationFinished(oldSession, succeeded = true))
        assertTrue(gate.initializationFinished(newSession, succeeded = true))
        assertTrue(gate.canSpeak)
    }

    private fun readyGate(): SpeechGate {
        val gate = SpeechGate()
        val session = gate.start()
        assertTrue(gate.initializationFinished(session, succeeded = true))
        return gate
    }
}
