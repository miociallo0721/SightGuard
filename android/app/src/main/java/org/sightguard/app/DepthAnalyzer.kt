package org.sightguard.app

import android.content.Context
import android.graphics.Bitmap
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/** Each output owns the exact upright source snapshot used by all its models. */
data class DepthSnapshot(val bitmap: Bitmap, val detections: List<Detection>, val evidence: DepthEvidence, val heatmap: Bitmap)

class DepthAnalyzer(context: Context, private val onResult: (DepthSnapshot) -> Unit, private val onStatus: (String) -> Unit) : AutoCloseable {
    private val worker = Executors.newSingleThreadExecutor()
    private val handler = Handler(Looper.getMainLooper())
    private val busy = AtomicBoolean(false)
    @Volatile private var closed = false
    @Volatile private var detector: DepthDetector? = null
    private var lastSubmittedAt = -2000L
    @Volatile private var intervalMs = 2000L
    init {
        worker.execute {
            try {
                val started = SystemClock.elapsedRealtime(); detector = DepthDetector(context.applicationContext)
                val load = SystemClock.elapsedRealtime() - started
                Log.i("SG_DEPTH", "loaded sha=${detector!!.modelSha} loadMs=$load pssKb=${android.os.Debug.getPss()}")
                handler.post { if (!closed) onStatus("Depth Anything V2 Small · 加载 ${load}ms · 低频 CPU") }
            } catch (error: Exception) {
                Log.e("SG_DEPTH", "load failed", error)
                handler.post { if (!closed) onStatus("UNKNOWN：Depth 加载失败 ${error.message}") }
            }
        }
    }
    @Synchronized fun offer(bitmap: Bitmap, key: SpatialFrameKey, detections: List<Detection>, force: Boolean = false) {
        val now = SystemClock.elapsedRealtime()
        if (closed || detector == null || (!force && now - lastSubmittedAt < intervalMs) || !busy.compareAndSet(false, true)) return
        val copy = try { bitmap.copy(Bitmap.Config.ARGB_8888, false) } catch (error: RuntimeException) { busy.set(false); return }
        lastSubmittedAt = now
        worker.execute {
            var posted = false
            try {
                if (closed) return@execute
                val result = detector!!.analyze(copy, key)
                intervalMs = maxOf(2000L, result.latencyMs * 3).coerceAtMost(8000)
                val snapshot = DepthSnapshot(copy, detections.toList(), result, result.heatmap())
                Log.i("SG_DEPTH", "frame=${key.frameIndex} sourceMs=${key.timestampMs} inferMs=${result.latencyMs} ageMs=${result.completedAtMs - key.capturedAtMs} quality=${result.quality} intervalMs=$intervalMs pssKb=${android.os.Debug.getPss()}")
                posted = true
                handler.post {
                    if (!closed) onResult(snapshot) else { snapshot.bitmap.recycle(); snapshot.heatmap.recycle() }
                }
            } catch (error: Exception) {
                Log.e("SG_DEPTH", "inference failed", error)
                handler.post { if (!closed) onStatus("UNKNOWN：Depth 推理失败") }
            } finally { if (!posted) copy.recycle(); busy.set(false) }
        }
    }
    @Synchronized override fun close() {
        if (closed) return
        closed = true
        worker.execute { detector?.close(); detector = null; Log.i("SG_DEPTH", "released pssKb=${android.os.Debug.getPss()}") }
        worker.shutdown()
    }
    companion object { const val RESULT_TTL_MS = 3000L }
}
