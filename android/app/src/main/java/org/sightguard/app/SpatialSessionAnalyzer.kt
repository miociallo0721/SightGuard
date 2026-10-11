package org.sightguard.app

import android.content.Context
import android.graphics.Bitmap
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

data class SpatialSettings(val depth: Boolean, val semantics: Boolean, val grid: Boolean = false)
data class SpatialSnapshot(val key: SpatialFrameKey, val bitmap: Bitmap, val detections: List<Detection>, val depth: DepthEvidence?, val heatmap: Bitmap?, val semantics: GroundEvidence?, val completedAtMs: Long, val grid: SpatialGridEvidence? = null)

/** YOLO remains on its own priority chain. Auxiliary models run sequentially on ONE source snapshot. */
class SpatialSessionAnalyzer(context: Context, private val settings: SpatialSettings, private val onResult: (SpatialSnapshot) -> Unit, private val onStatus: (String) -> Unit, private val motionProvider: (SpatialFrameKey) -> MotionEvidence = { MotionEvidence(MotionQuality.UNKNOWN,"IMU unavailable") }) : AutoCloseable {
    private val worker = Executors.newSingleThreadExecutor()
    private val handler = Handler(Looper.getMainLooper())
    private val busy = AtomicBoolean(false)
    @Volatile private var closed = false
    @Volatile private var ready = false
    private var depth: DepthDetector? = null
    private var semantics: SpatialDetector? = null
    private var lastSubmittedAt = -2000L
    @Volatile private var intervalMs = 2000L
    private val application = context.applicationContext
    private val yoloSha = org.json.JSONObject(context.assets.open("model_manifest.json").bufferedReader().use { it.readText() }).getString("onnx_sha256")
    @Volatile private var primarySlowFrames = 0
    fun notePrimaryLatency(latencyMs: Long) { primarySlowFrames = if (latencyMs > 180) (primarySlowFrames + 1).coerceAtMost(20) else (primarySlowFrames - 1).coerceAtLeast(0) }
    init {
        worker.execute {
            if (settings.depth && !closed) try {
                val start = SystemClock.elapsedRealtime(); depth = DepthDetector(context.applicationContext)
                Log.i("SG_DEPTH", "loaded sha=${depth!!.modelSha} loadMs=${SystemClock.elapsedRealtime()-start} pssKb=${android.os.Debug.getPss()}")
            } catch (error: Exception) { Log.e("SG_DEPTH", "load failed", error); status("UNKNOWN：Depth 加载失败 ${error.message}") }
              catch (error: OutOfMemoryError) { Log.e("SG_DEPTH", "load memory failure", error); status("UNKNOWN：Depth 加载内存不足") }
            if (settings.semantics && !closed) try {
                val start = SystemClock.elapsedRealtime(); semantics = SpatialDetector(context.applicationContext)
                Log.i("SG_SPATIAL", "loadMs=${SystemClock.elapsedRealtime()-start}")
            } catch (error: Exception) { Log.e("SG_SPATIAL", "load failed", error); status("UNKNOWN：语义加载失败 ${error.message}") }
              catch (error: OutOfMemoryError) { Log.e("SG_SPATIAL", "load memory failure", error); status("UNKNOWN：语义加载内存不足") }
            ready = true
            status("Depth ${if(depth!=null) "就绪" else if(settings.depth) "UNKNOWN" else "关闭"} · SegFormer ${if(semantics!=null) "就绪" else if(settings.semantics) "UNKNOWN" else "关闭"} · 同帧低频分析")
        }
    }
    private fun status(message: String) { handler.post { if (!closed) onStatus(message) } }
    @Synchronized fun offer(bitmap: Bitmap, key: SpatialFrameKey, detections: List<Detection>, force: Boolean = false) {
        val now = SystemClock.elapsedRealtime()
        if (closed || !ready || (!force && now-lastSubmittedAt < intervalMs) || !busy.compareAndSet(false,true)) return
        val memory = android.app.ActivityManager.MemoryInfo()
        (application.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager).getMemoryInfo(memory)
        val power = application.getSystemService(Context.POWER_SERVICE) as android.os.PowerManager
        val thermalLimited = android.os.Build.VERSION.SDK_INT >= 29 && power.currentThermalStatus >= android.os.PowerManager.THERMAL_STATUS_SEVERE
        if (memory.lowMemory || thermalLimited || primarySlowFrames >= 5) {
            intervalMs = maxOf(intervalMs,6000L)
            Log.w("SG_SPACE","degrade intervalMs=$intervalMs lowMemory=${memory.lowMemory} thermal=$thermalLimited slowFrames=$primarySlowFrames")
            if(memory.lowMemory || thermalLimited) { busy.set(false); status("UNKNOWN：内存或热状态受限，暂停辅助推理"); return }
        }
        val copy = try { bitmap.copy(Bitmap.Config.ARGB_8888,false) } catch (error: RuntimeException) { busy.set(false); status("UNKNOWN：快照内存不足"); return }
        lastSubmittedAt = now
        worker.execute {
            var posted = false
            try {
                if (closed) return@execute
                val d = try { depth?.analyze(copy,key) } catch(error: Exception) { Log.e("SG_DEPTH","infer failed",error); status("UNKNOWN：Depth 推理失败"); null } catch(error: OutOfMemoryError) { Log.e("SG_DEPTH","infer memory failure",error); status("UNKNOWN：Depth 推理内存不足"); null }
                if (closed) return@execute
                val s = try { semantics?.analyzeAll(copy,key) } catch(error: Exception) { Log.e("SG_SPATIAL","infer failed",error); status("UNKNOWN：语义推理失败"); null } catch(error: OutOfMemoryError) { Log.e("SG_SPATIAL","infer memory failure",error); status("UNKNOWN：语义推理内存不足"); null }
                val completed = SystemClock.elapsedRealtime()
                intervalMs = maxOf(2000L,(completed-now)*3).coerceAtMost(10000L)
                if(primarySlowFrames>=5) intervalMs=maxOf(intervalMs,6000L)
                val motion=motionProvider(key)
                val grid=if(settings.grid) {
                    val depthSource=if(d?.quality=="VALID") GridDepthSource(key,64,64,FloatArray(4096) { i -> d.sample((i%64+.5f)/64,(i/64+.5f)/64) },d.modelSha) else null
                    SpatialGridFusion().fuse(key,s?.let { GridSemanticSource(it.key,it.cells,SpatialDetector.MODEL_SHA) },depthSource,GridDetectionSource(key,detections,yoloSha),motion)
                } else null
                val snapshot = SpatialSnapshot(key,copy,detections.toList(),d,d?.heatmap(),s,completed,grid)
                if(grid!=null) Log.i("SG_GRID","frame=${key.frameIndex} sourceMs=${key.timestampMs} captureNs=${key.captureTimestampNs} clock=${key.timestampClock} rotation=${key.rotationDegrees} processedAtMs=$completed state=${grid.state} cells=${grid.cells.size} motion=${motion.quality} gyro=${motion.gyroRadPerSecond} synced=${motion.synchronized} samples=${motion.sampleCount} states=${grid.cells.groupingBy{it.state}.eachCount()} failures=${grid.failures} riskChanged=${grid.productionDecisionChanged}")
                Log.i("SG_SPACE","generation=${key.generation} epoch=${key.epoch} frame=${key.frameIndex} sourceMs=${key.timestampMs} size=${key.width}x${key.height} depthMs=${d?.latencyMs} semanticMs=${s?.latencyMs} ageMs=${completed-key.capturedAtMs} intervalMs=$intervalMs pssKb=${android.os.Debug.getPss()}")
                posted = true
                handler.post { if (!closed) onResult(snapshot) else { copy.recycle(); snapshot.heatmap?.recycle() } }
            } finally { if (!posted) copy.recycle(); busy.set(false) }
        }
    }
    override fun close() = release(null)
    @Synchronized fun release(onReleased: (() -> Unit)?) {
        if (closed) return
        closed = true
        worker.execute {
            depth?.close(); depth=null; semantics?.close(); semantics=null
            Log.i("SG_SPACE","released pssKb=${android.os.Debug.getPss()}")
            if(onReleased!=null) handler.post { onReleased() }
        }
        worker.shutdown()
    }
    companion object { const val RESULT_TTL_MS = 3000L }
}
