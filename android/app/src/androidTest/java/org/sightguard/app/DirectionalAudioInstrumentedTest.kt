package org.sightguard.app

import android.media.AudioAttributes
import android.media.MediaPlayer
import android.os.SystemClock
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/** Real local WAV playback probe. Capture each test externally with streamAudio.
 * This verifies the player path only; controller lifecycle tests remain separate.
 */
class DirectionalAudioInstrumentedTest {
    @Test fun localLeftPlayback() = play(Direction.LEFT, "alert_left_pedestrian")
    @Test fun localCenterPlayback() = play(Direction.CENTER, "alert_center_pedestrian")
    @Test fun localRightPlayback() = play(Direction.RIGHT, "alert_right_pedestrian")

    private fun play(direction: Direction, resource: String) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val started = CountDownLatch(1)
        val completed = CountDownLatch(1)
        val failure = AtomicReference<String?>(null)
        var player: MediaPlayer? = null
        val windowStart = SystemClock.elapsedRealtime()
        instrumentation.runOnMainSync {
            val active = MediaPlayer()
            player = active
            try {
                val id = context.resources.getIdentifier(resource, "raw", context.packageName)
                check(id != 0) { "Missing packaged WAV: $resource" }
                context.resources.openRawResourceFd(id).use {
                    active.setDataSource(it.fileDescriptor, it.startOffset, it.length)
                }
                active.setAudioAttributes(AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                active.setOnPreparedListener { prepared ->
                    try {
                        val mix = DirectionalAudio.mix(direction)
                        prepared.setVolume(mix.leftGain, mix.rightGain)
                        prepared.start()
                        check(prepared.isPlaying) { "MediaPlayer did not start" }
                        Log.i("SG_AUDIO_PROBE", "onStart direction=$direction resource=$resource")
                        started.countDown()
                    } catch (error: Exception) {
                        failure.set(error.javaClass.simpleName)
                        started.countDown(); completed.countDown()
                    }
                }
                active.setOnCompletionListener {
                    Log.i("SG_AUDIO_PROBE", "completed direction=$direction resource=$resource")
                    completed.countDown()
                }
                active.setOnErrorListener { _, what, extra ->
                    failure.set("MediaPlayer error $what/$extra")
                    started.countDown(); completed.countDown()
                    true
                }
                active.prepareAsync()
            } catch (error: Exception) {
                failure.set(error.javaClass.simpleName)
                started.countDown(); completed.countDown()
            }
        }
        try {
            assertTrue("No actual playback start", started.await(5, TimeUnit.SECONDS))
            assertTrue("Playback failed: ${failure.get()}", failure.get() == null)
            assertTrue("No real MediaPlayer completion", completed.await(5, TimeUnit.SECONDS))
            assertTrue("Playback failed: ${failure.get()}", failure.get() == null)
            // Keep one isolated ten-second capture window per direction.
            val remaining = 10_000L - (SystemClock.elapsedRealtime() - windowStart)
            if (remaining > 0) SystemClock.sleep(remaining)
        } finally {
            instrumentation.runOnMainSync { player?.release(); player = null }
        }
    }
}
