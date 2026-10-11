package org.sightguard.app

import android.content.Context
import android.media.AudioManager
import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/** Actual controller fallback playback; no fabricated playback callbacks. */
class SpeechControllerInstrumentedTest {
    @Test fun controllerLeftPlayback() = playback(Direction.LEFT, "左侧有行人")
    @Test fun controllerCenterPlayback() = playback(Direction.CENTER, "前方有行人")
    @Test fun controllerRightPlayback() = playback(Direction.RIGHT, "右侧有行人")

    private fun playback(direction: Direction, text: String) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val events = CopyOnWriteArrayList<SpeechPlaybackEvent>()
        val started = CountDownLatch(1)
        val completed = CountDownLatch(1)
        lateinit var controller: SpeechController
        val startTime = SystemClock.elapsedRealtime()
        instrumentation.runOnMainSync {
            reportCapabilities(context)
            controller = SpeechController(context, {}, {}, { event ->
                events += event
                if (event.phase == SpeechPlaybackPhase.ON_START) started.countDown()
                if (event.phase == SpeechPlaybackPhase.COMPLETED) completed.countDown()
            })
            controller.startLocalFallbackForTest()
            assertFalse(controller.canSpeakDynamic)
            assertTrue(controller.offer(text, false, "probe-$direction", Long.MAX_VALUE, direction))
        }
        try {
            assertTrue("No actual controller onStart", started.await(5, TimeUnit.SECONDS))
            assertTrue("No actual controller completion", completed.await(5, TimeUnit.SECONDS))
            val actual = events.single { it.phase == SpeechPlaybackPhase.ON_START }
            assertEquals(direction, actual.direction)
            assertEquals(text, actual.spokenText)
            assertEquals("local_media_player", actual.detail)
            assertEquals("probe-$direction", actual.eventId)
            assertTrue(events.none { it.phase == SpeechPlaybackPhase.CANCELLED || it.phase == SpeechPlaybackPhase.SUPPRESSED })
            val remaining = 10_000L - (SystemClock.elapsedRealtime() - startTime)
            if (remaining > 0) SystemClock.sleep(remaining)
        } finally {
            instrumentation.runOnMainSync { controller.close() }
        }
    }

    @Test fun dynamicOverviewIsNeverSubstitutedByGenericFallback() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val events = CopyOnWriteArrayList<SpeechPlaybackEvent>()
        instrumentation.runOnMainSync {
            val controller = SpeechController(instrumentation.targetContext, {}, {}, { events += it })
            try {
                controller.startLocalFallbackForTest()
                assertFalse(controller.offer("环境概览：左侧两名行人，右侧一辆汽车", true,
                    "overview", Long.MAX_VALUE, requireExactSpeech = true))
            } finally { controller.close() }
        }
        instrumentation.waitForIdleSync()
        assertTrue(events.any { it.phase == SpeechPlaybackPhase.SUPPRESSED && it.detail == "dynamic_speech_unavailable" })
        assertTrue(events.none { it.phase == SpeechPlaybackPhase.ON_START })
    }

    private fun reportCapabilities(context: Context) {
        val manager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val outputs = manager.getDevices(AudioManager.GET_DEVICES_OUTPUTS).map { it.type }
        if (Build.VERSION.SDK_INT >= 32) {
            val spatializer = manager.spatializer
            Log.i("SG_AUDIO_CAPABILITY", "api=${Build.VERSION.SDK_INT} available=${spatializer.isAvailable} " +
                "enabled=${spatializer.isEnabled} immersiveAudioLevel=${spatializer.immersiveAudioLevel} outputTypes=$outputs")
        } else {
            Log.i("SG_AUDIO_CAPABILITY", "api=${Build.VERSION.SDK_INT} spatializer=unsupported outputTypes=$outputs")
        }
        Log.i("SG_AUDIO_CAPABILITY", "backend=stereo_panning hrtfBackend=false headphoneLocalizationValidated=false")
    }
}
