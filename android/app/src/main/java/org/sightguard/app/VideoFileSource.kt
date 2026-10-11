package org.sightguard.app

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.provider.OpenableColumns

/** A synchronously accessed local-video decoder backed by [MediaMetadataRetriever].
 *
 * Calls to [frameAt] must be made serially on one analysis thread. Returned bitmap ownership
 * belongs to the caller. MediaMetadataRetriever applies the video's rotation metadata to its
 * bitmap, so this class does not rotate the decoded bitmap a second time.
 */
class VideoFileSource(context: Context, uri: Uri) : AutoCloseable {
    private val retriever = MediaMetadataRetriever()
    private var closed = false
    private var frameThread: Thread? = null

    val durationMs: Long
    /** Encoded track dimensions before applying [rotationDegrees]. */
    val width: Int
    /** Encoded track dimensions before applying [rotationDegrees]. */
    val height: Int
    val rotationDegrees: Int
    val name: String

    init {
        val appContext = context.applicationContext
        name = displayName(appContext, uri)
        try {
            retriever.setDataSource(appContext, uri)
            durationMs = requiredLong(MediaMetadataRetriever.METADATA_KEY_DURATION, "duration")
            width = requiredInt(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH, "width")
            height = requiredInt(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT, "height")
            val rawRotation = retriever.extractMetadata(
                MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION,
            )?.toIntOrNull() ?: 0
            rotationDegrees = rawRotation.takeIf { it >= 0 }
                ?.floorMod(360)
                ?.takeIf { it % 90 == 0 }
                ?: 0
            require(durationMs > 0L) { "Video duration must be positive" }
            require(width > 0 && height > 0) { "Video dimensions must be positive" }
        } catch (failure: Exception) {
            retriever.release()
            throw IllegalArgumentException("Cannot read video '$name'", failure)
        }
    }

    /**
     * Returns the closest frame the platform decoder can provide near [positionMs].
     *
     * `timestampMs` is the clamped request position, not a decoded presentation timestamp: the
     * MediaMetadataRetriever API does not expose the selected frame's PTS. Consumers must inspect
     * [VideoDecodedFrame.timestampSource] before using this value for temporal calculations.
     */
    @Synchronized
    fun frameAt(positionMs: Long): VideoDecodedFrame {
        check(!closed) { "Video source is closed" }
        val caller = Thread.currentThread()
        val previousCaller = frameThread
        check(previousCaller == null || previousCaller === caller) {
            "Video frames must be decoded serially on one thread"
        }
        frameThread = caller

        val clampedPositionMs = positionMs.coerceIn(0L, durationMs)
        val requestedUs = clampedPositionMs * 1_000L
        val bitmap = decodeFrame(requestedUs)
            ?: if (clampedPositionMs == durationMs && durationMs > 1L) {
                decodeFrame((durationMs - 1L) * 1_000L)
            } else {
                null
            }
            ?: throw IllegalStateException("No frame could be decoded from '$name' at ${clampedPositionMs}ms")

        return VideoDecodedFrame(
            bitmap = bitmap,
            timestampMs = clampedPositionMs,
            timestampSource = TIMESTAMP_SOURCE,
            requestedPositionMs = positionMs,
        )
    }

    private fun decodeFrame(timeUs: Long): Bitmap? {
        val requestedMaximumSide = maxOf(width, height).coerceAtMost(MAX_BITMAP_SIDE)
        val decoded = if (android.os.Build.VERSION.SDK_INT >= 27) {
            retriever.getScaledFrameAtTime(
                timeUs,
                MediaMetadataRetriever.OPTION_CLOSEST,
                requestedMaximumSide,
                requestedMaximumSide,
            )
        } else {
            retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST)
        } ?: return null

        if (decoded.width <= MAX_BITMAP_SIDE && decoded.height <= MAX_BITMAP_SIDE) return decoded

        val scale = MAX_BITMAP_SIDE.toFloat() / maxOf(decoded.width, decoded.height)
        val resized = Bitmap.createScaledBitmap(
            decoded,
            (decoded.width * scale).toInt().coerceAtLeast(1),
            (decoded.height * scale).toInt().coerceAtLeast(1),
            true,
        )
        if (resized !== decoded) decoded.recycle()
        return resized
    }

    private fun requiredLong(key: Int, label: String): Long =
        retriever.extractMetadata(key)?.toLongOrNull()
            ?: throw IllegalArgumentException("Video metadata has no valid $label")

    private fun requiredInt(key: Int, label: String): Int =
        retriever.extractMetadata(key)?.toIntOrNull()
            ?: throw IllegalArgumentException("Video metadata has no valid $label")

    @Synchronized
    override fun close() {
        if (!closed) {
            closed = true
            retriever.release()
        }
    }

    private fun displayName(context: Context, uri: Uri): String {
        val queriedName = runCatching {
            context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
                ?.use { cursor ->
                    if (cursor.moveToFirst()) {
                        val column = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                        if (column >= 0) cursor.getString(column) else null
                    } else {
                        null
                    }
                }
        }.getOrNull()
        return queriedName?.takeIf { it.isNotBlank() }
            ?: uri.lastPathSegment?.substringAfterLast('/')?.takeIf { it.isNotBlank() }
            ?: "video"
    }

    private fun Int.floorMod(modulus: Int): Int = ((this % modulus) + modulus) % modulus

    private companion object {
        const val MAX_BITMAP_SIDE = 640
        const val TIMESTAMP_SOURCE = "requested_position_ms_option_closest_no_pts"
    }
}

/** A decoded frame. The caller owns [bitmap] and should recycle it when no longer displayed. */
data class VideoDecodedFrame(
    val bitmap: Bitmap,
    /** Clamped request position; MediaMetadataRetriever does not expose the selected frame PTS. */
    val timestampMs: Long,
    val timestampSource: String,
    /** Original input passed to [VideoFileSource.frameAt], before duration clamping. */
    val requestedPositionMs: Long,
)
