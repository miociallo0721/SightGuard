package org.sightguard.app

import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.io.FileNotFoundException
import java.util.concurrent.atomic.AtomicReference
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class VideoFileSourceInstrumentedTest {
    @Test
    fun landscapePortraitAndRotationFixturesDecodeAtBoundedSizeAcrossSeekPositions() {
        val cases = listOf(
            FixtureCase("bus_landscape.mp4", encodedWidth = 640, encodedHeight = 360, rotation = 0, portrait = false),
            FixtureCase("bus_portrait.mp4", encodedWidth = 360, encodedHeight = 640, rotation = 0, portrait = true),
            FixtureCase("bus_rotation_90.mp4", encodedWidth = 640, encodedHeight = 360, rotation = 90, portrait = true),
        )

        for (case in cases) {
            val file = copyFixture(case.fileName)
            VideoFileSource(
                InstrumentationRegistry.getInstrumentation().targetContext,
                Uri.fromFile(file),
            ).use { source ->
                assertTrue("duration for ${case.fileName}", source.durationMs in 2_000L..4_000L)
                assertEquals("encoded width for ${case.fileName}", case.encodedWidth, source.width)
                assertEquals("encoded height for ${case.fileName}", case.encodedHeight, source.height)
                assertEquals("rotation metadata for ${case.fileName}", case.rotation, source.rotationDegrees)
                assertTrue("name for ${case.fileName}", source.name.contains(case.fileName))

                val positions = listOf(0L, source.durationMs / 2, 2_350L, 400L, source.durationMs)
                for (position in positions) {
                    val frame = source.frameAt(position)
                    try {
                        assertEquals(position, frame.requestedPositionMs)
                        assertEquals(position.coerceIn(0L, source.durationMs), frame.timestampMs)
                        assertEquals("requested_position_ms_option_closest_no_pts", frame.timestampSource)
                        assertTrue("width bound for ${case.fileName}", frame.bitmap.width in 1..640)
                        assertTrue("height bound for ${case.fileName}", frame.bitmap.height in 1..640)
                        assertEquals(
                            "upright display orientation for ${case.fileName}",
                            case.portrait,
                            frame.bitmap.height > frame.bitmap.width,
                        )
                    } finally {
                        frame.bitmap.recycle()
                    }
                }

                val outsideRange = source.frameAt(source.durationMs + 500L)
                try {
                    assertEquals(source.durationMs + 500L, outsideRange.requestedPositionMs)
                    assertEquals(source.durationMs, outsideRange.timestampMs)
                } finally {
                    outsideRange.bitmap.recycle()
                }

                val beforeStart = source.frameAt(-300L)
                try {
                    assertEquals(-300L, beforeStart.requestedPositionMs)
                    assertEquals(0L, beforeStart.timestampMs)
                } finally {
                    beforeStart.bitmap.recycle()
                }
            }
        }
    }

    @Test
    fun invalidSourceAndClosedSourceFailClearly() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val invalid = File(context.cacheDir, "video-source-invalid.mp4")
        invalid.writeText("not an mp4")
        try {
            var openFailure: Throwable? = null
            try {
                VideoFileSource(context, Uri.fromFile(invalid))
            } catch (failure: Throwable) {
                openFailure = failure
            }
            assertNotNull("invalid media should fail to open", openFailure)
            assertTrue(openFailure is IllegalArgumentException)

            val source = VideoFileSource(context, Uri.fromFile(copyFixture("bus_landscape.mp4")))
            source.close()
            var decodeFailure: Throwable? = null
            try {
                source.frameAt(0L)
            } catch (failure: Throwable) {
                decodeFailure = failure
            }
            assertTrue("closed source should reject decode", decodeFailure is IllegalStateException)
        } finally {
            invalid.delete()
        }
    }

    @Test
    fun decoderRejectsCallsFromASecondThread() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val source = VideoFileSource(context, Uri.fromFile(copyFixture("bus_landscape.mp4")))
        try {
            source.frameAt(0L).bitmap.recycle()
            val otherThreadFailure = AtomicReference<Throwable?>()
            val otherThread = Thread {
                try {
                    source.frameAt(100L).bitmap.recycle()
                } catch (failure: Throwable) {
                    otherThreadFailure.set(failure)
                }
            }
            otherThread.start()
            otherThread.join()
            assertTrue(otherThreadFailure.get() is IllegalStateException)
        } finally {
            source.close()
        }
    }

    private fun copyFixture(fileName: String): File {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val output = File(instrumentation.targetContext.cacheDir, fileName)
        try {
            instrumentation.context.assets.open("video-fixtures/$fileName").use { input ->
                output.outputStream().use { outputStream -> input.copyTo(outputStream) }
            }
        } catch (missing: FileNotFoundException) {
            throw AssertionError(
                "Missing local video fixture '$fileName'; run scripts/generate_android_video_fixtures.py before androidTest.",
                missing,
            )
        }
        return output
    }

    private data class FixtureCase(
        val fileName: String,
        val encodedWidth: Int,
        val encodedHeight: Int,
        val rotation: Int,
        val portrait: Boolean,
    )
}
