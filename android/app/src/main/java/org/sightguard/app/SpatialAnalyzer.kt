package org.sightguard.app

import ai.onnxruntime.*
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import java.nio.FloatBuffer
import java.security.MessageDigest
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.exp
import kotlin.math.roundToInt

data class SpatialFrameKey(val generation: Long, val epoch: Long, val frameIndex: Int, val timestampMs: Double, val width: Int, val height: Int, val capturedAtMs: Long)
data class SemanticCell(val left: Float, val top: Float, val right: Float, val bottom: Float, val classId: Int, val confidence: Float)
data class GroundEvidence(val key: SpatialFrameKey, val cells: List<SemanticCell>, val latencyMs: Long, val completedAtMs: Long, val state: String)
data class SpatialInput(val values: FloatArray, val resizedWidth: Int, val resizedHeight: Int, val padLeft: Int, val padTop: Int)

/** Real ADE20K semantics only. Own CPU session and thread; never enters risk scoring. */
class SpatialDetector(context: Context) : AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    init {
        val bytes = context.assets.open(MODEL_FILE).use { it.readBytes() }
        val sha = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
        require(sha == MODEL_SHA) { "SegFormer SHA mismatch" }
        OrtSession.SessionOptions().use { options ->
            options.setIntraOpNumThreads(1); options.setInterOpNumThreads(1)
            session = environment.createSession(bytes, options)
        }
        try {
            require((session.inputInfo.getValue("pixel_values").info as TensorInfo).shape.contentEquals(longArrayOf(1, 3, 256, 256)))
            require((session.outputInfo.getValue("class_logits").info as TensorInfo).shape.contentEquals(longArrayOf(1, 150, 64, 64)))
        } catch (error: Exception) { session.close(); throw error }
        Log.i("SG_SPATIAL", "loaded real SegFormer MiT-B0 sha=$sha input=[1,3,256,256] output=[1,150,64,64] threads=1")
    }
    fun prepare(bitmap: Bitmap): SpatialInput {
        val ratio = minOf(256f / bitmap.width, 256f / bitmap.height)
        val width = (bitmap.width * ratio).roundToInt().coerceAtLeast(1); val height = (bitmap.height * ratio).roundToInt().coerceAtLeast(1)
        val dx = (256 - width) / 2; val dy = (256 - height) / 2
        val scaled = Bitmap.createScaledBitmap(bitmap, width, height, true)
        val pixels = IntArray(width * height); scaled.getPixels(pixels, 0, width, 0, 0, width, height)
        if (scaled !== bitmap) scaled.recycle()
        val values = FloatArray(3 * 256 * 256) // Mean-colored padding becomes zero after normalization.
        val mean = floatArrayOf(.485f, .456f, .406f); val std = floatArrayOf(.229f, .224f, .225f)
        pixels.forEachIndexed { i, pixel ->
            val offset = (dy + i / width) * 256 + dx + i % width
            values[offset] = (Color.red(pixel) / 255f - mean[0]) / std[0]
            values[offset + 65536] = (Color.green(pixel) / 255f - mean[1]) / std[1]
            values[offset + 131072] = (Color.blue(pixel) / 255f - mean[2]) / std[2]
        }
        return SpatialInput(values, width, height, dx, dy)
    }
    fun infer(values: FloatArray): Array<FloatArray> {
        require(values.size == 3 * 256 * 256)
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(values), longArrayOf(1, 3, 256, 256)).use { input ->
            session.run(mapOf("pixel_values" to input)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val classes = (result[0].value as Array<Array<Array<FloatArray>>>)[0]
                return Array(150) { cls -> FloatArray(4096) { pixel -> classes[cls][pixel / 64][pixel % 64] } }
            }
        }
    }
    fun analyze(bitmap: Bitmap, key: SpatialFrameKey): GroundEvidence {
        val started = SystemClock.elapsedRealtime(); val input = prepare(bitmap); val logits = infer(input.values)
        val cells = mutableListOf<SemanticCell>()
        for (pixel in 0 until 4096) {
            var best = 0
            for (cls in 1 until 150) if (logits[cls][pixel] > logits[best][pixel]) best = cls
            if (best !in GROUND_CLASSES) continue
            val max = logits[best][pixel]; var denominator = 0.0
            for (cls in 0 until 150) denominator += exp((logits[cls][pixel] - max).toDouble())
            val confidence = (1.0 / denominator).toFloat()
            if (confidence < MIN_CONFIDENCE) continue
            val left = ((pixel % 64 * 4f - input.padLeft) / input.resizedWidth).coerceIn(0f, 1f)
            val top = ((pixel / 64 * 4f - input.padTop) / input.resizedHeight).coerceIn(0f, 1f)
            val right = (((pixel % 64 + 1) * 4f - input.padLeft) / input.resizedWidth).coerceIn(0f, 1f)
            val bottom = (((pixel / 64 + 1) * 4f - input.padTop) / input.resizedHeight).coerceIn(0f, 1f)
            if (right > left && bottom > top) cells += SemanticCell(left, top, right, bottom, best, confidence)
        }
        val completed = SystemClock.elapsedRealtime()
        val evidence = GroundEvidence(key, cells, completed - started, completed, if (cells.isEmpty()) "ABSTAIN：缺少可信地面语义" else "地面语义 · 非安全区域")
        Log.i("SG_SPATIAL", "frame=${key.frameIndex} sourceMs=${key.timestampMs} cells=${cells.size} latencyMs=${evidence.latencyMs} evidenceAgeMs=${completed - key.capturedAtMs} state=${evidence.state}")
        return evidence
    }
    override fun close() { session.close() }
    companion object {
        const val MODEL_FILE = "segformer_mit_b0_ade20k_256.onnx"
        const val MODEL_SHA = "85913f7adae1b787da8cbe4a046e6d8a22bae901340d548be2fea46ec7ff2e0f"
        const val MIN_CONFIDENCE = .35f
        const val RESULT_TTL_MS = 500L
        const val SAMPLE_INTERVAL_MS = 2000L
        val GROUND_CLASSES = setOf(3, 6, 11)
    }
}

/** One in-flight snapshot, independently stoppable, no analysis backlog. */
class SpatialAnalyzer(context: Context, private val onResult: (GroundEvidence) -> Unit, private val onStatus: (String) -> Unit) : AutoCloseable {
    private val worker = Executors.newSingleThreadExecutor()
    private val handler = Handler(Looper.getMainLooper())
    private val busy = AtomicBoolean(false)
    @Volatile private var closed = false
    @Volatile private var detector: SpatialDetector? = null
    private var lastSubmittedAtMs = -SpatialDetector.SAMPLE_INTERVAL_MS
    init {
        worker.execute {
            try {
                val started = SystemClock.elapsedRealtime(); detector = SpatialDetector(context.applicationContext)
                val loadMs = SystemClock.elapsedRealtime() - started
                handler.post { if (!closed) onStatus("SegFormer MiT-B0 · 低频CPU · 加载 ${loadMs}ms") }
                Log.i("SG_SPATIAL", "loadMs=$loadMs")
            } catch (error: Exception) {
                Log.e("SG_SPATIAL", "load failed", error); handler.post { if (!closed) onStatus("UNKNOWN：空间模型不可用 ${error.message}") }
            }
        }
    }
    @Synchronized fun offer(bitmap: Bitmap, key: SpatialFrameKey, force: Boolean = false) {
        val now = SystemClock.elapsedRealtime()
        if (closed || detector == null || (!force && now - lastSubmittedAtMs < SpatialDetector.SAMPLE_INTERVAL_MS) || !busy.compareAndSet(false, true)) return
        val copy = try { bitmap.copy(Bitmap.Config.ARGB_8888, false) } catch (error: RuntimeException) { busy.set(false); return }
        lastSubmittedAtMs = now
        worker.execute {
            try {
                if (closed) return@execute
                val result = detector!!.analyze(copy, key)
                handler.post { if (!closed) onResult(result) }
            } catch (error: Exception) { Log.e("SG_SPATIAL", "inference failed", error); handler.post { if (!closed) onStatus("UNKNOWN：空间推理失败") } }
            finally { copy.recycle(); busy.set(false) }
        }
    }
    @Synchronized override fun close() {
        if (closed) return
        closed = true
        worker.execute { detector?.close(); detector = null; Log.i("SG_SPATIAL", "released") }
        worker.shutdown()
    }
}
