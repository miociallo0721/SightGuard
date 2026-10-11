package org.sightguard.app

import ai.onnxruntime.*
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.os.SystemClock
import org.json.JSONObject
import java.nio.FloatBuffer
import java.security.MessageDigest
import java.io.File

data class DepthEvidence(val key: SpatialFrameKey, val values: FloatArray, val size: Int, val transform: SquareTransform, val latencyMs: Long, val completedAtMs: Long, val modelSha: String, val quality: String) {
    fun sample(x: Float, y: Float): Float {
        val mx = transform.modelX(x).toInt().coerceIn(0, size - 1)
        val my = transform.modelY(y).toInt().coerceIn(0, size - 1)
        return values[my * size + mx]
    }
    fun sourceValues(): FloatArray {
        val output = FloatArray(transform.resizedWidth * transform.resizedHeight)
        for (y in 0 until transform.resizedHeight) for (x in 0 until transform.resizedWidth)
            output[y * transform.resizedWidth + x] = values[(y + transform.padTop) * size + x + transform.padLeft]
        return output
    }
    /** Inset bbox proxy, never an instance mask. */
    fun region(d: Detection): DepthStatistics {
        val left = d.left + (d.right - d.left) * .15f; val right = d.right - (d.right - d.left) * .15f
        val top = d.top + (d.bottom - d.top) * .15f; val bottom = d.bottom - (d.bottom - d.top) * .15f
        val samples = FloatArray(24 * 24) { i -> sample(left + (right - left) * (i % 24 + .5f) / 24, top + (bottom - top) * (i / 24 + .5f) / 24) }
        return RelativeDepthRules.summarize(samples, sourceValues())
    }
    fun heatmap(): Bitmap {
        val scene = sourceValues().filter { it.isFinite() && it >= 0 }.sorted()
        val low = if (scene.isEmpty()) 0f else RelativeDepthRules.quantile(scene, .01f)
        val high = if (scene.isEmpty()) 0f else RelativeDepthRules.quantile(scene, .99f)
        val pixels = IntArray(values.size) { i ->
            val value = values[i]
            if (!value.isFinite() || value < 0 || high <= low || quality != "VALID") Color.TRANSPARENT
            else {
                val fraction = ((value - low) / (high - low)).coerceIn(0f, 1f)
                Color.HSVToColor(floatArrayOf((1 - fraction) * 240, .9f, 1f))
            }
        }
        val full = Bitmap.createBitmap(pixels, size, size, Bitmap.Config.ARGB_8888)
        val crop = Bitmap.createBitmap(full, transform.padLeft, transform.padTop, transform.resizedWidth, transform.resizedHeight)
        if (crop !== full) full.recycle()
        return crop
    }
}

/** Single-owner CPU adapter using the existing PC checkpoint; larger output is relatively nearer. */
class DepthDetector(context: Context) : AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    val modelSha: String
    init {
        val manifest = JSONObject(context.assets.open("depth_manifest.json").bufferedReader().use { it.readText() })
        val file = File(context.noBackupFilesDir, MODEL_FILE)
        fun hash(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { stream ->
                val buffer = ByteArray(32768)
                while (true) { val count = stream.read(buffer); if (count < 0) break; digest.update(buffer, 0, count) }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }
        val expected = manifest.getString("onnx_sha256")
        if (!file.isFile || hash(file) != expected) {
            val pending = File(context.noBackupFilesDir, "$MODEL_FILE.pending")
            try {
                context.assets.open(MODEL_FILE).use { source -> pending.outputStream().use { source.copyTo(it,32768) } }
                require(hash(pending) == expected) { "Depth SHA mismatch" }
                check(pending.renameTo(file)) { "Depth local model extraction failed" }
            } finally { pending.delete() }
        }
        modelSha = hash(file)
        require(modelSha == manifest.getString("onnx_sha256")) { "Depth SHA mismatch" }
        OrtSession.SessionOptions().use { options ->
            options.setIntraOpNumThreads(1); options.setInterOpNumThreads(1)
            session = environment.createSession(file.absolutePath, options)
        }
        try {
            require((session.inputInfo.getValue("pixel_values").info as TensorInfo).shape.contentEquals(longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())))
            require((session.outputInfo.getValue("predicted_depth").info as TensorInfo).shape.contentEquals(longArrayOf(1, SIZE.toLong(), SIZE.toLong())))
        } catch (error: Exception) { session.close(); throw error }
    }
    fun prepare(bitmap: Bitmap): Pair<FloatArray, SquareTransform> {
        val transform = SquareTransform(bitmap.width, bitmap.height, SIZE)
        val scaled = Bitmap.createScaledBitmap(bitmap, transform.resizedWidth, transform.resizedHeight, true)
        val pixels = IntArray(scaled.width * scaled.height); scaled.getPixels(pixels, 0, scaled.width, 0, 0, scaled.width, scaled.height)
        if (scaled !== bitmap) scaled.recycle()
        val values = FloatArray(3 * SIZE * SIZE)
        pixels.forEachIndexed { i, pixel ->
            val offset = (transform.padTop + i / transform.resizedWidth) * SIZE + transform.padLeft + i % transform.resizedWidth
            values[offset] = (Color.red(pixel) / 255f - .485f) / .229f
            values[offset + SIZE * SIZE] = (Color.green(pixel) / 255f - .456f) / .224f
            values[offset + 2 * SIZE * SIZE] = (Color.blue(pixel) / 255f - .406f) / .225f
        }
        return values to transform
    }
    fun infer(values: FloatArray): FloatArray {
        require(values.size == 3 * SIZE * SIZE)
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(values), longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())).use { input ->
            session.run(mapOf("pixel_values" to input)).use { result ->
                @Suppress("UNCHECKED_CAST") val rows = (result[0].value as Array<Array<FloatArray>>)[0]
                return FloatArray(SIZE * SIZE) { i -> rows[i / SIZE][i % SIZE] }
            }
        }
    }
    fun analyze(bitmap: Bitmap, key: SpatialFrameKey): DepthEvidence {
        require(bitmap.width == key.width && bitmap.height == key.height)
        val started = SystemClock.elapsedRealtime(); val (tensor, transform) = prepare(bitmap); val values = infer(tensor)
        val completed = SystemClock.elapsedRealtime()
        val draft = DepthEvidence(key, values, SIZE, transform, completed - started, completed, modelSha, "UNKNOWN")
        val stats = RelativeDepthRules.summarize(draft.sourceValues(), draft.sourceValues())
        val quality = if (stats.validFraction >= .8f && stats.framePercentile != null) "VALID" else "UNKNOWN"
        return draft.copy(quality = quality)
    }
    override fun close() { session.close() }
    companion object { const val SIZE = 280; const val MODEL_FILE = "depth_anything_v2_small_280.onnx" }
}
