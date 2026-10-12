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

data class SpatialFrameKey(val generation: Long, val epoch: Long, val frameIndex: Int, val timestampMs: Double, val width: Int, val height: Int, val capturedAtMs: Long, val captureTimestampNs: Long? = null, val timestampClock: String = "UNKNOWN", val rotationDegrees: Int = 0)
data class SemanticCell(val left: Float, val top: Float, val right: Float, val bottom: Float, val classId: Int, val confidence: Float)
data class GroundEvidence(val key: SpatialFrameKey, val cells: List<SemanticCell>, val latencyMs: Long, val completedAtMs: Long, val state: String)
data class SpatialInput(val values: FloatArray, val resizedWidth: Int, val resizedHeight: Int, val padLeft: Int, val padTop: Int)

/** Real ADE20K semantics only. Own CPU session and thread; never enters risk scoring. */
class SpatialDetector(context: Context, private val intraOpThreads: Int = 1) : AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    init {
        val mapping = org.json.JSONObject(context.assets.open("ade20k_labels.json").bufferedReader().use { it.readText() }).getJSONArray("labels")
        require(mapping.length() == 150 && (0 until 150).all { mapping.getJSONObject(it).getInt("id") == it }) { "ADE20K mapping mismatch" }
        val expected = mapOf(0 to "wall", 53 to "stairs", 59 to "stairway", 121 to "step", 93 to "pole", 42 to "column", 38 to "railing", 95 to "bannister", 14 to "door", 32 to "fence", 4 to "tree", 87 to "streetlight", 96 to "escalator", 21 to "water", 34 to "rock")
        require(expected.all { (id,label) -> mapping.getJSONObject(id).getString("rawLabel")==label }) { "Environment ADE20K class mapping mismatch" }
        val bytes = context.assets.open(MODEL_FILE).use { it.readBytes() }
        val sha = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
        require(sha == MODEL_SHA) { "SegFormer SHA mismatch" }
        OrtSession.SessionOptions().use { options ->
            require(intraOpThreads in 1..2)
            options.setIntraOpNumThreads(intraOpThreads); options.setInterOpNumThreads(1)
            session = environment.createSession(bytes, options)
        }
        try {
            require((session.inputInfo.getValue("pixel_values").info as TensorInfo).shape.contentEquals(longArrayOf(1, 3, 256, 256)))
            require((session.outputInfo.getValue("class_logits").info as TensorInfo).shape.contentEquals(longArrayOf(1, 150, 64, 64)))
        } catch (error: Exception) { session.close(); throw error }
        Log.i("SG_SPATIAL", "loaded real SegFormer MiT-B0 sha=$sha input=[1,3,256,256] output=[1,150,64,64] threads=$intraOpThreads")
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
                val tensor=result[0] as OnnxTensor
                val flat=FloatArray(150*4096)
                tensor.floatBuffer.get(flat)
                return Array(150) {cls -> flat.copyOfRange(cls*4096,(cls+1)*4096)}
            }
        }
    }
    fun analyze(bitmap: Bitmap, key: SpatialFrameKey): GroundEvidence = analyzeClasses(bitmap, key, GROUND_CLASSES)
    fun analyzeAll(bitmap: Bitmap, key: SpatialFrameKey): GroundEvidence = analyzeClasses(bitmap, key, (0 until 150).toSet())
    private fun analyzeClasses(bitmap: Bitmap, key: SpatialFrameKey, allowedClasses: Set<Int>): GroundEvidence {
        val started = SystemClock.elapsedRealtime(); val input = prepare(bitmap); val prepared=SystemClock.elapsedRealtime(); val logits = infer(input.values);val inferred=SystemClock.elapsedRealtime()
        val cells = mutableListOf<SemanticCell>()
        // Traverse contiguous class planes rather than jumping between 150 distant
        // arrays per pixel. Keep the exact argmax and softmax accumulation order.
        val best=IntArray(4096);val maxima=logits[0].copyOf()
        for(cls in 1 until 150) {
            val scores=logits[cls]
            for(pixel in 0 until 4096) if(scores[pixel]>maxima[pixel]) {maxima[pixel]=scores[pixel];best[pixel]=cls}
        }
        val eligible=BooleanArray(4096) {pixel ->
            val x=pixel%64*4;val y=pixel/64*4
            best[pixel] in allowedClasses && maxima[pixel].isFinite() && x+4>input.padLeft && x<input.padLeft+input.resizedWidth && y+4>input.padTop && y<input.padTop+input.resizedHeight
        }
        val denominators=DoubleArray(4096)
        for(cls in 0 until 150) {
            val scores=logits[cls]
            for(pixel in 0 until 4096) if(eligible[pixel]) {
                denominators[pixel]+=exp((scores[pixel]-maxima[pixel]).toDouble())
                // Positive terms cannot recover a pixel already below the confidence gate.
                if((1.0/denominators[pixel]).toFloat()<MIN_CONFIDENCE) eligible[pixel]=false
            }
        }
        for (pixel in 0 until 4096) {
            if(!eligible[pixel]) continue
            val confidence=(1.0/denominators[pixel]).toFloat()
            if(!confidence.isFinite() || confidence<MIN_CONFIDENCE) continue
            val left = ((pixel % 64 * 4f - input.padLeft) / input.resizedWidth).coerceIn(0f, 1f)
            val top = ((pixel / 64 * 4f - input.padTop) / input.resizedHeight).coerceIn(0f, 1f)
            val right = (((pixel % 64 + 1) * 4f - input.padLeft) / input.resizedWidth).coerceIn(0f, 1f)
            val bottom = (((pixel / 64 + 1) * 4f - input.padTop) / input.resizedHeight).coerceIn(0f, 1f)
            if (right > left && bottom > top) cells += SemanticCell(left, top, right, bottom, best[pixel], confidence)
        }
        val completed = SystemClock.elapsedRealtime()
        val evidence = GroundEvidence(key, cells, completed - started, completed, if (cells.isEmpty()) "ABSTAIN：缺少可信语义" else "真实语义 · 非安全区域")
        Log.i("SG_SPATIAL", "frame=${key.frameIndex} sourceMs=${key.timestampMs} cells=${cells.size} prepareMs=${prepared-started} inferMs=${inferred-prepared} postMs=${completed-inferred} latencyMs=${evidence.latencyMs} evidenceAgeMs=${completed - key.capturedAtMs} state=${evidence.state}")
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
