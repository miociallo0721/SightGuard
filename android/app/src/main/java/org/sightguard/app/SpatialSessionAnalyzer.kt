package org.sightguard.app

import android.content.Context
import android.graphics.Bitmap
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

data class SpatialSettings(val depth: Boolean, val semantics: Boolean, val grid: Boolean = false, val ignoreThermalForTest: Boolean = false)
data class SpatialSnapshot(val key: SpatialFrameKey, val bitmap: Bitmap, val detections: List<Detection>, val depth: DepthEvidence?, val heatmap: Bitmap?, val semantics: GroundEvidence?, val completedAtMs: Long, val grid: SpatialGridEvidence? = null)

/** YOLO remains on its own priority chain. Auxiliary models run sequentially on ONE source snapshot. */
class SpatialSessionAnalyzer(context: Context, private val settings: SpatialSettings, private val onResult: (SpatialSnapshot) -> Unit, private val onStatus: (String) -> Unit, private val motionProvider: (SpatialFrameKey) -> MotionEvidence = { MotionEvidence(MotionQuality.UNKNOWN,"IMU unavailable") }, private val onSemantic: (GroundEvidence, Boolean, MotionEvidence) -> Unit = { _, _, _ -> }, private val onSceneChanged: () -> Unit = {}, private val onShadow: (WallApproachRecord) -> Unit = {}, private val motionIntervalProvider: (SpatialFrameKey,SpatialFrameKey?) -> MotionEvidence = { key,_ -> motionProvider(key) }) : AutoCloseable {
    private val worker = Executors.newSingleThreadExecutor()
    private val handler = Handler(Looper.getMainLooper())
    private val busy = AtomicBoolean(false)
    @Volatile private var closed = false
    @Volatile private var ready = false
    private var depth: DepthDetector? = null
    private var semantics: SpatialDetector? = null
    private val environmentConfig = loadEnvironmentConfig(context)
    private val shadow = WallShadowAnalyzer(loadWallResearchConfig(context))
    private var lastDepthAt = -6000L
    @Volatile private var latestSignature: FloatArray = floatArrayOf()
    @Volatile private var latestKey: SpatialFrameKey? = null
    @Volatile private var acceptedSignature: FloatArray? = null
    private var lastSubmittedAt = -2000L
    @Volatile private var intervalMs = environmentConfig.sampleIntervalMs
    private val application = context.applicationContext
    private val ignoreThermalForTest = settings.ignoreThermalForTest &&
        application.applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE != 0
    private val yoloSha = org.json.JSONObject(context.assets.open("model_manifest.json").bufferedReader().use { it.readText() }).getString("onnx_sha256")
    @Volatile private var primarySlowFrames = 0
    fun notePrimaryLatency(latencyMs: Long) { primarySlowFrames = if (latencyMs > 180) (primarySlowFrames + 1).coerceAtMost(20) else (primarySlowFrames - 1).coerceAtLeast(0) }
    init {
        if(ignoreThermalForTest) Log.w("SG_SPACE","debug thermal gate override enabled; system protection unchanged")
        worker.execute {
            if (settings.semantics && !closed) try {
                val start = SystemClock.elapsedRealtime(); semantics = SpatialDetector(context.applicationContext)
                Log.i("SG_SPATIAL", "loadMs=${SystemClock.elapsedRealtime()-start}")
            } catch (error: Exception) { Log.e("SG_SPATIAL", "load failed", error); status("UNKNOWN：语义加载失败 ${error.message}") }
              catch (error: OutOfMemoryError) { Log.e("SG_SPATIAL", "load memory failure", error); status("UNKNOWN：语义加载内存不足") }
            ready = true
            status("Depth ${if(settings.depth) "按需加载" else "关闭"} · SegFormer ${if(semantics!=null) "就绪" else if(settings.semantics) "UNKNOWN" else "关闭"} · 同帧低频分析")
        }
    }
    private fun status(message: String) { handler.post { if (!closed) onStatus(message) } }
    @Synchronized fun offer(bitmap: Bitmap, key: SpatialFrameKey, detections: List<Detection>, force: Boolean = false) {
        val now = SystemClock.elapsedRealtime()
        if (closed) return
        latestSignature = semanticSceneSignature(bitmap); latestKey = key
        acceptedSignature?.let { previous ->
            if(!SemanticSceneContinuity.matches(previous,latestSignature,environmentConfig.maxSceneDifference)) {
                acceptedSignature=null
                handler.post { if(!closed) onSceneChanged() }
            }
        }
        if (closed || !ready || (!force && now-lastSubmittedAt < intervalMs) || !busy.compareAndSet(false,true)) return
        // A rejected attempt also consumes the interval. Thermal/memory pauses must not
        // poll services and emit status/log messages on every primary CameraX frame.
        lastSubmittedAt = now
        val memory = android.app.ActivityManager.MemoryInfo()
        (application.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager).getMemoryInfo(memory)
        val power = application.getSystemService(Context.POWER_SERVICE) as android.os.PowerManager
        val thermalLevel=if(android.os.Build.VERSION.SDK_INT>=29) power.currentThermalStatus else 0
        val plan=SpatialResourcePolicy.plan(if(ignoreThermalForTest) 0 else thermalLevel,memory.lowMemory,primarySlowFrames,environmentConfig.sampleIntervalMs)
        if(plan.reason!="normal" || ignoreThermalForTest) Log.w("SG_SPACE","resource intervalMs=${plan.intervalMs} thermalLevel=$thermalLevel thermalOverride=$ignoreThermalForTest reason=${plan.reason} slowFrames=$primarySlowFrames")
        if(plan.paused) { intervalMs=plan.intervalMs;busy.set(false);status("UNKNOWN：内存、热状态或主链负载受限，暂停辅助推理");return }
        val copy = try { bitmap.copy(Bitmap.Config.ARGB_8888,false) } catch (error: RuntimeException) { busy.set(false); status("UNKNOWN：快照内存不足"); return }
        worker.execute {
            var posted = false
            try {
                if (closed) return@execute
                if (closed) return@execute
                val s = try { semantics?.analyzeAll(copy,key) } catch(error: Exception) { Log.e("SG_SPATIAL","infer failed",error); status("UNKNOWN：语义推理失败"); null } catch(error: OutOfMemoryError) { Log.e("SG_SPATIAL","infer memory failure",error); status("UNKNOWN：语义推理内存不足"); null }
                val sourceSignature = semanticSceneSignature(copy)
                if (s != null) handler.post {
                    val latest=latestKey
                    val current=latest != null && latest.generation==key.generation && latest.epoch==key.epoch &&
                        latest.width==key.width && latest.height==key.height &&
                        SemanticSceneContinuity.matches(sourceSignature,latestSignature,environmentConfig.maxSceneDifference)
                    if(!closed) { if(current) acceptedSignature=sourceSignature; onSemantic(s,current,motionProvider(key)) }
                }
                val currentThermal=if(android.os.Build.VERSION.SDK_INT>=29) power.currentThermalStatus else 0
                val useDepth=settings.depth && plan.depthAllowed && (ignoreThermalForTest || currentThermal<2) && now-lastDepthAt>=6000 && primarySlowFrames==0
                if(useDepth) lastDepthAt=now
                if(useDepth && depth==null && !closed) try {
                    val start=SystemClock.elapsedRealtime();depth=DepthDetector(application)
                    Log.i("SG_DEPTH","loaded sha=${depth!!.modelSha} loadMs=${SystemClock.elapsedRealtime()-start} pssKb=${android.os.Debug.getPss()}")
                } catch(error: Exception) {Log.e("SG_DEPTH","load failed",error);status("UNKNOWN：Depth 加载失败 ${error.message}")}
                  catch(error: OutOfMemoryError) {Log.e("SG_DEPTH","load memory failure",error);status("UNKNOWN：Depth 加载内存不足")}
                val d = try { if(useDepth) depth?.analyze(copy,key) else null } catch(error: Exception) { Log.e("SG_DEPTH","infer failed",error); status("UNKNOWN：Depth 推理失败"); null } catch(error: OutOfMemoryError) { Log.e("SG_DEPTH","infer memory failure",error); status("UNKNOWN：Depth 推理内存不足"); null }
                val completed = SystemClock.elapsedRealtime()
                intervalMs = maxOf(plan.intervalMs,(s?.latencyMs ?: (completed-now))*3).coerceAtMost(10000L)
                if(s!=null) {
                    val latest=latestKey
                    val current=latest!=null && latest.generation==key.generation && latest.epoch==key.epoch && SemanticSceneContinuity.matches(sourceSignature,latestSignature,environmentConfig.maxSceneDifference)
                    val record=shadow.analyze(copy,s,d,motionIntervalProvider(key,shadow.previousKey),current)
                    Log.i("SG_WALL_SHADOW",wallRecordJson(record).toString())
                    handler.post {if(!closed) onShadow(record)}
                }
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
