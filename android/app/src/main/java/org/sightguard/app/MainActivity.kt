package org.sightguard.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Paint
import android.net.Uri
import android.os.*
import android.util.Log
import android.util.Rational
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import org.json.JSONArray
import org.json.JSONObject
import org.sightguard.app.core.EventState
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

private data class FrameResult(val vision: VisionResult, val width: Int, val height: Int, val receivedAtMs: Long)
private data class PendingSpeech(val token: Long, val epoch: Long, val stage: VisionStage, val presentation: PresentedAlert, val result: VisionResult, val requestKey: String)

class MainActivity : ComponentActivity() {
    private lateinit var autoPolicy: AutoStartPolicy
    private var autoStart by mutableStateOf(true)
    private var permissionExplanation by mutableStateOf(false)
    private var pendingPermissionStart = false
    private var manualPermission = false
    private var lastSpokenAtMs by mutableLongStateOf(0)
    private var displayNow by mutableLongStateOf(0)
    private var running by mutableStateOf(false)
    private var status by mutableStateOf("待启动")
    private var inputStatus by mutableStateOf("摄像头未启动")
    private var speechStatus by mutableStateOf("开始后自动启用离线中文提醒")
    private var lastSpeech by mutableStateOf("尚未播放提醒")
    private var lastEvent by mutableStateOf("尚无已确认事件")
    private var muted by mutableStateOf(false)
    private var boxesEnabled by mutableStateOf(true)
    private var riskEnabled by mutableStateOf(true)
    private var depthEnabled by mutableStateOf(true)
    private var baselineDetector by mutableStateOf(false)
    private var metricResearchEnabled by mutableStateOf(false)
    private var metricResearchStatus by mutableStateOf("米制墙距研究关闭 · 未进行手机卷尺量测")
    private var spatialEnabled by mutableStateOf(true)
    private var ignoreAuxThermalForTest = false
    private lateinit var environmentEngine: SemanticEnvironmentEngine
    private var environmentStatus by mutableStateOf("墙面：等待有效语义观测")
    private val semanticPending = mutableSetOf<String>()
    private val environmentSpeechDeduplicator = EnvironmentSpeechDeduplicator()
    private var lastRapidAtMs = -1L
    private var gridEnabled by mutableStateOf(false)
    private var imuEnabled by mutableStateOf(true)
    @Volatile private var imuMonitor: ImuMonitor? = null
    @Volatile private var cameraTimestampClock = "UNKNOWN"
    private var spatialPanel by mutableStateOf(false)
    private var settingsPanel by mutableStateOf(false)
    private val diagnostics=AwarenessDiagnostics()
    private val exportDocument=registerForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { uri ->
        if(uri!=null) {
            val data=diagnostics.snapshot(sessionDiagnostics())
            Thread {
                runCatching { contentResolver.openOutputStream(uri)?.bufferedWriter()?.use { it.write(data) } }
                    .onSuccess { handler.post { speechStatus="诊断已导出" } }
                    .onFailure { error -> handler.post { speechStatus="导出失败：${error.message}" } }
            }.start()
        }
    }
    private var spaceView by mutableStateOf("原始画面")
    private var semanticLayer by mutableStateOf("All Semantics")
    private var layerAlpha by mutableFloatStateOf(.4f)
    private var spatialStatus by mutableStateOf("空间模型关闭")
    private var environmentOverview by mutableStateOf("当前环境信息不足。")
    private var overviewSpeechNote by mutableStateOf("")
    private var wallShadowStatus by mutableStateOf("ABSTAIN：等待研究证据")
    private var spatialSnapshot by mutableStateOf<SpatialSnapshot?>(null)
    @Volatile private var spatialSession: SpatialSessionAnalyzer? = null
    private var spatialReleasePending = false
    private var videoMode by mutableStateOf(false)
    private var videoUri: Uri? = null
    private var videoName by mutableStateOf("尚未选择视频")
    private var videoDuration by mutableLongStateOf(0)
    private var videoPosition by mutableLongStateOf(0)
    private var videoPlaying by mutableStateOf(false)
    private var videoBitmap by mutableStateOf<Bitmap?>(null)
    private var frame by mutableStateOf<FrameResult?>(null)
    private var fps by mutableFloatStateOf(0f)
    private var latencyMs by mutableLongStateOf(0)
    private var lastValidAtMs = 0L
    private lateinit var previewView: PreviewView
    private lateinit var speech: SpeechController
    private val handler = Handler(Looper.getMainLooper())
    private var provider: ProcessCameraProvider? = null
    private var analysis: ImageAnalysis? = null
    private var cameraState: androidx.lifecycle.LiveData<CameraState>? = null
    private var worker: ExecutorService? = null
    private var primaryReleasePending = false
    private var restartAfterRelease = false
    private var releaseSession: (() -> Unit)? = null
    private var stage: VisionStage? = null
    private var videoSource: VideoFileSource? = null
    private var videoBusy = false
    private var nextVideoPosition = 0L
    private var videoRateStart = 0L
    private var videoRateFrames = 0
    private var videoEpoch = 0L
    private var processedVideoEpoch = -1L
    private var startedAtMs = 0L
    private var interrupted = false
    private val pendingSpeech = linkedMapOf<String, PendingSpeech>()
    private var speechSequence = 0L
    @Volatile private var generation = 0L
    @Volatile private var active = false
    private val permission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        Log.i("SG_AUTO","permission granted=$granted foreground=${autoPolicy.foreground} enabled=$autoStart")
        permissionExplanation = false
        if (granted) {
            pendingPermissionStart = !autoPolicy.explicitlyStopped && (manualPermission || autoStart)
            if(autoPolicy.permissionGranted(manualPermission)) { pendingPermissionStart=false; startPerception() }
        } else { pendingPermissionStart=false; status = "摄像头权限被拒绝，请在系统设置中授权" }
    }
    private val videoPicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            stopPerception(false)
            runCatching { contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION) }
            videoUri = uri; videoMode = true; videoPosition = 0; videoDuration = 0
            videoName = uri.lastPathSegment?.substringAfterLast('/') ?: "本地视频"
            status = "视频已选择，点击开始分析"; inputStatus = "文件本地解码 · 不上传"
            Log.i("SG_VIDEO", "SAF selected scheme=${uri.scheme}")
        }
    }
    private val watchdog = object : Runnable {
        override fun run() {
            displayNow = SystemClock.elapsedRealtime()
            if(running) serviceEnvironment()
            val snapshot = spatialSnapshot
            if (snapshot != null && !(videoMode && !videoPlaying && frame?.vision?.records?.frameIndex == snapshot.key.frameIndex) && SystemClock.elapsedRealtime() - snapshot.key.capturedAtMs > SpatialSessionAnalyzer.RESULT_TTL_MS) {
                spatialSnapshot = null; environmentOverview = "当前环境信息不足。"
            }
            if (running && !videoMode) {
                if(imuMonitor?.currentDeviceMotion()?.quality==MotionQuality.RAPID) {
                    spatialSnapshot=null; environmentOverview="当前环境信息不足。"; spatialStatus="UNKNOWN：设备快速转动，空间快照暂停显示；精确帧同步 $cameraTimestampClock"
                }
                val now = SystemClock.elapsedRealtime()
                if (frame?.let { !ResultFreshness.isCurrent(it.receivedAtMs, now) } == true) { frame = null; fps = 0f; environmentOverview = "当前环境信息不足。" }
                val gap = now - (lastValidAtMs.takeIf { it > 0 } ?: startedAtMs)
                if (gap > (if (lastValidAtMs > 0) 3500 else 10000)) {
                    if (!interrupted) {
                        interrupted = true; status = "视觉感知暂时中断"
                        speech.offer("视觉感知暂时中断", true, "watchdog-$generation", now + 2000)
                        Log.w("SG_SESSION", "stale gapMs=$gap")
                    }
                }
            }
            handler.postDelayed(this, 300)
        }
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ignoreAuxThermalForTest = applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE != 0 &&
            intent.getBooleanExtra("ignore_aux_thermal_for_test",false)
        baselineDetector = getSharedPreferences("perception", MODE_PRIVATE).getBoolean("baseline_detector",false)
        environmentEngine = SemanticEnvironmentEngine(loadEnvironmentConfig(this))
        autoStart = getSharedPreferences("perception", MODE_PRIVATE).getBoolean("auto_start", true)
        // Debug fixture host keeps the target UID foreground without camera/model overlap.
        // It never edits the user's saved auto-start setting and cannot affect release builds.
        if(BuildConfig.DEBUG && intent.getBooleanExtra("foreground_fixture_test",false)) autoStart=false
        muted = getSharedPreferences("perception", MODE_PRIVATE).getBoolean("muted", false)
        videoMode = savedInstanceState?.getBoolean("video_mode") ?: false
        videoUri = savedInstanceState?.getString("video_uri")?.let(Uri::parse)
        videoPosition = savedInstanceState?.getLong("video_position") ?: 0
        videoName = savedInstanceState?.getString("video_name") ?: "尚未选择视频"
        autoPolicy = AutoStartPolicy(autoStart, savedInstanceState?.getBoolean("explicit_stop") ?: false).apply { fileMode=videoMode }
        previewView = PreviewView(this).apply { implementationMode = PreviewView.ImplementationMode.COMPATIBLE; scaleType = PreviewView.ScaleType.FIT_CENTER }
        speech = SpeechController(this, { speechStatus = it }, { lastSpeech = it; lastSpokenAtMs = SystemClock.elapsedRealtime() }, ::onPlayback)
        setContent { MaterialTheme(colorScheme = darkColorScheme(primary = Color(0xFF74DFCB), background = Color(0xFF090E17), surface = Color(0xFF17212F))) { Screen() } }
        handler.post(watchdog)
    }
    private fun requestStart() {
        autoPolicy.manualStart(); manualPermission = true
        if (videoMode && videoUri == null) { autoPolicy.sourceChanged(true); videoPicker.launch(arrayOf("video/*")); return }
        if (videoMode || checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startPerception()
        else permissionExplanation = true
    }
    private fun startPerception() {
        if (running || isFinishing || isDestroyed || !lifecycle.currentState.isAtLeast(androidx.lifecycle.Lifecycle.State.STARTED)) return
        if(primaryReleasePending) { restartAfterRelease=true; status="正在释放旧会话，随后启动…"; return }
        val token = ++generation
        environmentEngine.reset(); semanticPending.clear(); environmentSpeechDeduplicator.reset(); lastRapidAtMs=-1L
        running = true; active = true; frame = null; videoBitmap = null; fps = 0f; latencyMs = 0; lastValidAtMs = 0
        interrupted = false; pendingSpeech.clear(); startedAtMs = SystemClock.elapsedRealtime(); videoRateStart = 0; videoRateFrames = 0
        status = if(baselineDetector) "正在加载本地 YOLO26n…" else "正在加载本地 YOLOE…"; inputStatus = if (videoMode) "读取本地视频" else "等待后置摄像头"
        lastEvent = "尚无已确认事件"; videoPlaying = videoMode; videoBusy = false; processedVideoEpoch = -1; videoEpoch++
        speech.setMuted(muted); speech.start()
        val executor = Executors.newSingleThreadExecutor(); worker = executor
        var ownedStage: VisionStage? = null; var ownedVideo: VideoFileSource? = null
        releaseSession = { ownedVideo?.close(); ownedVideo = null; ownedStage?.close(); ownedStage = null }
        val selectedUri = videoUri; val isVideo = videoMode
        executor.execute {
            try {
                ownedStage = VisionStage(applicationContext, if (isVideo) "local_video" else "camera")
                if (isVideo) ownedVideo = VideoFileSource(applicationContext, selectedUri!!)
                handler.post {
                    if (!running || generation != token) return@post
                    stage = ownedStage; videoSource = ownedVideo
                    if (spatialEnabled || depthEnabled || gridEnabled || metricResearchEnabled) startSpatialSession()
                    if (isVideo) {
                        videoName = ownedVideo!!.name; videoDuration = ownedVideo!!.durationMs
                        videoPosition = videoPosition.coerceIn(0, (videoDuration - 1).coerceAtLeast(0))
                        nextVideoPosition = videoPosition
                        inputStatus = "本地解码 · 请求位置估计时间 · CPU"; pumpVideo(token)
                    } else bindCamera(token, executor, ownedStage!!)
                }
            } catch (error: Exception) { Log.e("SG_MODEL", "session load failed", error); handler.post { if (generation == token) fail("输入或模型加载失败：${error.message}") } }
        }
        Log.i("SG_SESSION", "start generation=$token source=${if (isVideo) "video" else "CameraX"} thermalOverride=$ignoreAuxThermalForTest")
    }
    @Suppress("DEPRECATION")
    @androidx.annotation.OptIn(markerClass = [androidx.camera.camera2.interop.ExperimentalCamera2Interop::class])
    private fun bindCamera(token: Long, executor: ExecutorService, vision: VisionStage) {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            if (!running || generation != token) return@addListener
            try {
                val cameraProvider = future.get(); provider = cameraProvider
                require(cameraProvider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA)) { "没有可用的后置摄像头" }
                val rotation = previewView.display?.rotation ?: android.view.Surface.ROTATION_0
                val preview = Preview.Builder().setTargetAspectRatio(AspectRatio.RATIO_4_3).setTargetRotation(rotation).build()
                preview.surfaceProvider = previewView.surfaceProvider
                val analyzer = ImageAnalysis.Builder().setTargetAspectRatio(AspectRatio.RATIO_4_3).setTargetRotation(rotation)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888).setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()
                analysis = analyzer
                var firstTimestampNs: Long? = null; var previousTimestampMs = -1L; var completed = 0
                var rateStart = SystemClock.elapsedRealtime(); var rateFrames = 0
                val trace = if (applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE != 0) File(filesDir, "parity_camera_trace.jsonl").apply {
                    writeText(JSONObject().put("source", "real CameraX ImageProxy").put("model_sha256", vision.modelSha).toString() + "\n")
                } else null
                analyzer.setAnalyzer(executor) { image ->
                    val receivedAt = SystemClock.elapsedRealtime()
                    try {
                        if (!active || generation != token) return@setAnalyzer
                        val sensor = image.imageInfo.timestamp
                        val capturedAt = if(cameraTimestampClock=="REALTIME") sensor/1_000_000 else receivedAt
                        if (firstTimestampNs == null) firstTimestampNs = sensor
                        val timestamp = (sensor - firstTimestampNs!!) / 1_000_000
                        if (timestamp <= previousTimestampMs) return@setAnalyzer
                        previousTimestampMs = timestamp
                        val bitmap = uprightBitmap(image); val width = bitmap.width; val height = bitmap.height
                        val result = try {
                            vision.process(bitmap, timestamp, "monotonic").also { v ->
                                spatialSession?.offer(bitmap, SpatialFrameKey(token, 0, v.records.frameIndex, v.records.timestampMs, width, height, capturedAt, sensor, cameraTimestampClock, image.imageInfo.rotationDegrees), v.raw)
                            }
                        } finally { bitmap.recycle() }
                        val completedAt = SystemClock.elapsedRealtime(); val latency = completedAt - receivedAt
                        spatialSession?.notePrimaryLatency(latency)
                        if (!ResultFreshness.isCurrent(capturedAt, completedAt)) { Log.w("SG_FRAME", "discard stale latencyMs=$latency"); return@setAnalyzer }
                        if (trace != null && completed < 200) trace.appendText(recordJson(result).toString() + "\n")
                        completed++; rateFrames++
                        val rate = rateFrames * 1000f / (completedAt - rateStart).coerceAtLeast(1)
                        if (completedAt - rateStart >= 3000) { rateStart = completedAt; rateFrames = 0 }
                        if (completed % 10 == 1) Log.i("SG_FRAME", "frame=$completed sensorMs=$timestamp size=${width}x$height objects=${result.raw.size} latencyMs=$latency yoloMs=${result.detectionLatencyMs} sourceAgeMs=${completedAt-capturedAt} fps=$rate pssKb=${android.os.Debug.getPss()} activeEvents=${result.risk.frame.events.count { it.state == EventState.ACTIVE }}")
                        handler.post {
                            if (!running || generation != token || !ResultFreshness.isCurrent(capturedAt, SystemClock.elapsedRealtime())) return@post
                            publish(result, width, height, capturedAt, latency, rate, vision, token, 0)
                        }
                    } catch (error: Exception) { Log.e("SG_FRAME", "inference failed", error); handler.post { if (running && generation == token) fail("视觉感知失败：${error.message}") } }
                    finally { image.close() }
                }
                val viewport = previewView.viewPort ?: ViewPort.Builder(Rational(3, 4), rotation).setScaleType(ViewPort.FIT).build()
                val camera = cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, UseCaseGroup.Builder().setViewPort(viewport).addUseCase(preview).addUseCase(analyzer).build())
                val timestampSource = androidx.camera.camera2.interop.Camera2CameraInfo.from(camera.cameraInfo).getCameraCharacteristic(android.hardware.camera2.CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)
                cameraTimestampClock = if(timestampSource == android.hardware.camera2.CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME) "REALTIME" else "UNKNOWN"
                Log.i("SG_IMU","cameraTimestampSource=$timestampSource clock=$cameraTimestampClock")
                cameraState = camera.cameraInfo.cameraState
                cameraState?.observe(this) { state -> if (running && generation == token) { inputStatus = "摄像头：${state.type}"; if (state.error != null) fail("摄像头连接失败（${state.error?.code}）") } }
                status = "等待有效检测帧"; Log.i("SG_CAMERA", "bound rear generation=$token")
            } catch (error: Exception) { Log.e("SG_CAMERA", "bind failed", error); fail("摄像头连接失败：${error.message}") }
        }, ContextCompat.getMainExecutor(this))
    }
    /** At most one decode/inference job. A seek replaces the request rather than queuing frames. */
    private fun pumpVideo(token: Long, snapshot: Boolean = false) {
        if (!running || generation != token || videoBusy || (!videoPlaying && !snapshot)) return
        val source = videoSource ?: return; val vision = stage ?: return; val executor = worker ?: return
        val epoch = videoEpoch; val position = nextVideoPosition; val receivedAt = SystemClock.elapsedRealtime()
        videoBusy = true
        executor.execute {
            var decoded: Bitmap? = null
            try {
                if (!active || generation != token) return@execute
                if (processedVideoEpoch != epoch) { vision.reset(); processedVideoEpoch = epoch }
                val video = source.frameAt(position); decoded = video.bitmap
                val result = vision.process(video.bitmap, video.timestampMs, "fps_fallback")
                spatialSession?.offer(video.bitmap, SpatialFrameKey(token, epoch, result.records.frameIndex, result.records.timestampMs, video.bitmap.width, video.bitmap.height, receivedAt), result.raw)
                val completedAt = SystemClock.elapsedRealtime(); val latency = completedAt - receivedAt
                val image = decoded; decoded = null
                handler.post {
                    if (!running || generation != token) { image?.recycle(); return@post }
                    videoBusy = false
                    if (videoEpoch != epoch) { image?.recycle(); pumpVideo(token, snapshot = true); return@post }
                    videoBitmap = image
                    videoPosition = video.timestampMs
                    if (videoRateStart == 0L) videoRateStart = receivedAt
                    videoRateFrames++
                    val actualRate = videoRateFrames * 1000f / (completedAt - videoRateStart).coerceAtLeast(1)
                    if (completedAt - videoRateStart >= 3000) { videoRateStart = completedAt; videoRateFrames = 0 }
                    publish(result, image!!.width, image.height, completedAt, latency, actualRate, vision, token, epoch)
                    Log.i("SG_VIDEO", "frame positionMs=$position requestedTimestampMs=${video.timestampMs} timeSource=${video.timestampSource} size=${image.width}x${image.height} objects=${result.raw.size} latencyMs=$latency epoch=$epoch")
                    nextVideoPosition = (position + latency.coerceAtLeast(100)).coerceAtMost(videoDuration - 1)
                    if (videoPlaying) {
                        if (position >= videoDuration - 1) { videoPlaying = false; status = "视频分析结束"; speech.cancelAlerts("video_ended") }
                        else { handler.postDelayed({ pumpVideo(token) }, (nextVideoPosition - position - latency).coerceAtLeast(0)) }
                    }
                }
            } catch (error: Exception) { Log.e("SG_VIDEO", "decode/infer failed", error); handler.post { if (generation == token) fail("视频分析失败：${error.message}") } }
            finally { decoded?.recycle() }
        }
    }
    private fun publish(result: VisionResult, width: Int, height: Int, receivedAt: Long, latency: Long, rate: Float, vision: VisionStage, token: Long, epoch: Long) {
        if(result.records.frameIndex%30==0) diagnostics.record("performance",JSONObject().put("source",if(videoMode) "video" else "camera").put("fps",rate).put("frame_latency_ms",latency).put("yolo_ms",result.detectionLatencyMs).put("pss_kb",android.os.Debug.getPss()))
        frame = FrameResult(result, width, height, receivedAt); lastValidAtMs = SystemClock.elapsedRealtime(); latencyMs = latency; fps = rate
        if(result.records.frameIndex % 10 == 0) Log.i("SG_RISK", "frame=${result.records.frameIndex} raw=${result.raw.size} eligible=${result.relevance.count{it.eligible}} contextual=${result.relevance.count{it.category==RelevanceCategory.CONTEXTUAL}} quality=${result.risk.frame.perception.status} levels=${result.risk.frame.events.filter{it.observedNow}.map{it.assessment?.level}}")
        status = if (videoMode && !videoPlaying) "视频已暂停" else "感知运行中"
        if (interrupted) { speech.offer("感知已恢复", true, "recovered-$token", SystemClock.elapsedRealtime() + 2000); interrupted = false }
        if (!speech.isReady || muted || (videoMode && !videoPlaying)) return
        val now = SystemClock.elapsedRealtime()
        val waitingEvents = pendingSpeech.values.flatMap { (it.presentation.merged + it.presentation.alert).map { a -> a.eventId } }.toSet()
        val proposal = vision.presenter.choose(result.risk.copy(alerts = result.risk.alerts.filter { it.eventId !in waitingEvents }), now) ?: return
        if (!proposal.important && !speech.canQueueOrdinaryAlert) return
        if (pendingSpeech.size >= 2 && !proposal.important) return
        lastEvent = "${proposal.alert.eventId} · ${proposal.alert.alertType} · ${proposal.text}"
        Log.i("SG_ALERT", "raw event=${proposal.alert.eventId} type=${proposal.alert.alertType} sourceMs=${proposal.alert.timestampMs} receivedAtMs=$receivedAt confirmedAtMs=$now direction=${proposal.direction} score=${proposal.alert.riskScore} text=${proposal.text}")
        val key = "$token:$epoch:${proposal.alert.eventId}:${proposal.alert.alertType}:${++speechSequence}"
        pendingSpeech[key] = PendingSpeech(token, epoch, vision, proposal, result, key)
        if (!speech.offer(proposal.text, proposal.important, key, vision.presenter.deadline(now), direction = proposal.direction)) pendingSpeech.remove(key)
    }
    private fun onPlayback(event: SpeechPlaybackEvent) {
        diagnostics.record("audio",JSONObject().put("phase",event.phase.logName).put("event_id",event.eventId).put("requested_text",event.requestedText).put("spoken_text",event.spokenText).put("deadline_ms",event.validUntilElapsedMs).put("detail",event.detail).put("resource",event.resourceName))
        if(event.eventId?.startsWith("semantic:")==true) {
            if(event.phase==SpeechPlaybackPhase.ON_START) {
                environmentEngine.playbackStarted(event.eventId)
                Log.i("SG_ENV", "onStart event=${event.eventId} text=${event.spokenText} resource=${event.resourceName}")
            }
            if(event.phase in setOf(SpeechPlaybackPhase.ON_START,SpeechPlaybackPhase.SUPPRESSED,SpeechPlaybackPhase.CANCELLED)) semanticPending.remove(event.eventId)
            return
        }
        val pending = pendingSpeech[event.eventId] ?: return
        when (event.phase) {
            SpeechPlaybackPhase.ON_START -> {
                pendingSpeech.remove(event.eventId)
                if (running && generation == pending.token && (!videoMode || videoEpoch == pending.epoch)) {
                    val spokenIds = (pending.presentation.merged + pending.presentation.alert).map { it.eventId }.toSet()
                    pending.result.risk.frame.observations.filter { it.eventId in spokenIds }.forEach { observation ->
                        observation.features?.spatial?.bboxNormalized?.let { box ->
                            environmentSpeechDeduplicator.yoloStarted(pending.token,pending.epoch,observation.classId,box,event.elapsedRealtimeMs,event.spokenText.orEmpty())
                        }
                    }
                    pending.stage.presenter.accepted(pending.presentation, pending.result.risk, event.elapsedRealtimeMs)
                    worker?.execute { if (active && generation == pending.token) pending.stage.accept(pending.presentation) }
                }
            }
            SpeechPlaybackPhase.SUPPRESSED, SpeechPlaybackPhase.CANCELLED -> pendingSpeech.remove(event.eventId)
            else -> Unit
        }
    }
    private fun seekVideo(position: Long) {
        environmentEngine.reset(); semanticPending.clear(); environmentSpeechDeduplicator.reset()
        spatialSnapshot = null; environmentOverview = "当前环境信息不足。"
        videoEpoch++; pendingSpeech.clear(); speech.cancelAlerts("video_seek")
        videoPosition = position.coerceIn(0, (videoDuration - 1).coerceAtLeast(0)); frame = null; videoBitmap = null; fps = 0f
        nextVideoPosition = videoPosition
        videoRateStart = 0; videoRateFrames = 0
        status = "跳转中，重建事件状态"; Log.i("SG_VIDEO", "seek positionMs=$videoPosition epoch=$videoEpoch clearedOverlay=true"); pumpVideo(generation, snapshot = true)
    }
    private fun togglePlayback() {
        if (!running) { requestStart(); return }
        videoPlaying = !videoPlaying
        if (videoPlaying) {
            videoRateStart = 0; videoRateFrames = 0; status = "恢复文件分析"
            if (videoPosition >= videoDuration - 1) seekVideo(0) else pumpVideo(generation)
        }
        else { pendingSpeech.clear(); speech.cancelAlerts("video_paused"); status = "视频已暂停" }
    }
    private fun fail(message: String) {
        autoPolicy.failed()
        val ready = speech.isReady
        stopPerception(false, !ready); status = message
        if (!ready) return
        speech.offer(if (message.startsWith("摄像头")) "摄像头连接失败" else "视觉感知暂时中断", true, "failure-$generation", SystemClock.elapsedRealtime() + 2500)
        handler.postDelayed({ if (!running) speech.stop(false) }, 3000)
    }
    private fun stopPerception(announce: Boolean = true, stopSpeech: Boolean = true) {
        restartAfterRelease = false
        if (!running) { if (stopSpeech) speech.stop(false); return }
        environmentEngine.reset(); semanticPending.clear(); environmentSpeechDeduplicator.reset(); environmentStatus="墙面：感知已停止"
        metricResearchStatus="米制研究：感知已停止，旧结果不再有效"
        running = false; active = false; generation++; pendingSpeech.clear(); videoPlaying = false; videoEpoch++
        spatialSession?.let { old -> spatialReleasePending=true; old.release { spatialReleasePending=false; if(running) startSpatialSession() } }; spatialSession = null; spatialSnapshot = null; spatialStatus = "空间资源已释放"; environmentOverview = "当前环境信息不足。"
        imuMonitor?.close(); imuMonitor = null; cameraTimestampClock = "UNKNOWN"
        cameraState?.removeObservers(this); cameraState = null; analysis?.clearAnalyzer(); analysis = null; provider?.unbindAll(); provider = null
        val release = releaseSession; releaseSession = null
        worker?.let { executor ->
            primaryReleasePending=true
            executor.execute { try { release?.invoke() } finally {
                Log.i("SG_SESSION", "model and input released")
                handler.post { primaryReleasePending=false; val restart=restartAfterRelease; restartAfterRelease=false; if(restart && autoPolicy.foreground && !autoPolicy.explicitlyStopped) startPerception() }
            } }; executor.shutdown()
        }
        worker = null; stage = null; videoSource = null; videoBusy = false
        frame = null; videoBitmap = null; fps = 0f; latencyMs = 0; lastValidAtMs = 0
        inputStatus = if (videoMode) "文件解码器已释放" else "摄像头已释放"; status = "感知已停止"
        if (stopSpeech) speech.stop(announce)
        Log.i("SG_SESSION", "stop input unbound generation=$generation")
    }
    override fun onResume() {
        super.onResume()
        val action = autoPolicy.enter(checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
        Log.i("SG_AUTO","foreground action=$action enabled=$autoStart stopped=${autoPolicy.explicitlyStopped} source=${if(videoMode) "video" else "camera"}")
        if(pendingPermissionStart && autoPolicy.permissionGranted(manualPermission)) { pendingPermissionStart=false; startPerception(); return }
        when (action) {
            "START" -> startPerception()
            "PERMISSION" -> { manualPermission = false; permissionExplanation = true }
        }
    }
    override fun onSaveInstanceState(outState: Bundle) { outState.putBoolean("explicit_stop", autoPolicy.explicitlyStopped); outState.putBoolean("video_mode",videoMode); outState.putString("video_uri",videoUri?.toString()); outState.putLong("video_position",videoPosition); outState.putString("video_name",videoName); super.onSaveInstanceState(outState) }
    override fun onStop() { autoPolicy.leave(running, restartAfterRelease); stopPerception(false); super.onStop() }
    override fun onConfigurationChanged(newConfig: android.content.res.Configuration) {
        super.onConfigurationChanged(newConfig)
        // The existing release barrier serializes model ownership when CameraX geometry changes.
        if(running) { stopPerception(false); startPerception() }
    }
    override fun onDestroy() { stopPerception(false); handler.removeCallbacksAndMessages(null); speech.close(); super.onDestroy() }

    private fun startSpatialSession() {
        environmentEngine.reset(); semanticPending.clear(); environmentSpeechDeduplicator.reset(); speech.cancelSemanticExcept(emptySet(),"semantic_settings_changed")
        environmentStatus=if(spatialEnabled) "墙面：等待有效语义观测" else "墙面：语义提示关闭"
        spatialSnapshot = null; environmentOverview = "当前环境信息不足。"
        imuMonitor?.close(); imuMonitor = null
        if(spatialReleasePending) return
        val previous = spatialSession; spatialSession = null
        if(previous!=null) {
            spatialReleasePending = true; spatialStatus = "正在释放旧空间会话…"
            previous.release { spatialReleasePending=false; if(running) startSpatialSession() }
            return
        }
        if (!running || !(spatialEnabled || depthEnabled || gridEnabled || metricResearchEnabled)) { spatialStatus = "空间模型关闭"; return }
        if(imuEnabled && !videoMode) imuMonitor = ImuMonitor(this)
        spatialStatus = "正在加载本地空间模型…"
        val token = generation
        spatialSession = SpatialSessionAnalyzer(this, SpatialSettings(depthEnabled, spatialEnabled, gridEnabled,ignoreAuxThermalForTest,metricResearchEnabled,1, stage?.modelSha ?: "unavailable"), { result ->
            val key = result.key
            if (running && generation == key.generation && (!videoMode || videoEpoch == key.epoch)) {
                val pausedExact = videoMode && !videoPlaying && frame?.vision?.records?.frameIndex == key.frameIndex
                if ((pausedExact || SystemClock.elapsedRealtime() - key.capturedAtMs <= SpatialSessionAnalyzer.RESULT_TTL_MS) && imuMonitor?.currentDeviceMotion()?.quality != MotionQuality.RAPID) spatialSnapshot = result
                else { result.bitmap.recycle(); result.heatmap?.recycle() }
                spatialStatus = "源帧 #${key.frameIndex} · Depth ${result.depth?.latencyMs ?: "—"}ms · 语义 ${result.semantics?.latencyMs ?: "—"}ms"
            } else { result.bitmap.recycle(); result.heatmap?.recycle() }
        }, { state ->
            if (running && generation == token) {
                spatialStatus = state
                val bitmap = videoBitmap; val records = frame?.vision?.records
                // Only the one model-ready notification requests an initial paused snapshot.
                // Error/thermal status must not recursively force another rejected request.
                if (state.endsWith("同帧低频分析") && videoMode && !videoPlaying && bitmap != null && records != null) spatialSession?.offer(bitmap, SpatialFrameKey(token, videoEpoch, records.frameIndex, records.timestampMs, bitmap.width, bitmap.height, SystemClock.elapsedRealtime()), frame!!.vision.raw, force = true)
            }
        }, { key -> imuMonitor?.evidence(key) ?: MotionEvidence(MotionQuality.UNKNOWN, if(videoMode) "video not synchronized to device IMU" else "IMU disabled") }, ::onSemanticEvidence, {
            if(running && generation==token) {
                environmentEngine.invalidate(SystemClock.elapsedRealtime(),"墙面：画面变化，等待重新确认")
                environmentSpeechDeduplicator.reset()
                speech.cancelSemanticExcept(emptySet(),"semantic_scene_changed")
                environmentStatus=environmentEngine.status
            }
        }, { record ->
            if(running && generation==record.key.generation && (!videoMode || videoEpoch==record.key.epoch)) {
                wallShadowStatus="${record.state} · ${record.abstainReason ?: "研究候选"} · features ${record.flow?.inliers ?: 0} · ${record.flowLatencyMs}ms"
                diagnostics.record("wall_shadow",wallRecordJson(record))
            }
        }, { key,previous -> imuMonitor?.evidence(key,windowStartNs=previous?.captureTimestampNs) ?: MotionEvidence(MotionQuality.UNKNOWN,"source not synchronized to device IMU") }, {record->
            if(running && generation==record.key.generation && (!videoMode || videoEpoch==record.key.epoch)) {
                metricResearchStatus="最近米制研究记录：${record.status} · 预测光轴深度 ${record.median?.let {"%.2f m".format(it)} ?: "UNKNOWN"} · 完成时年龄 ${record.sourceAgeMs}ms · ${record.rejectionReasons.joinToString()} · 未标定"
                diagnostics.record("wall_metric",metricWallJson(record))
            }
        }, { imuMonitor?.currentDeviceMotion() ?: MotionEvidence(MotionQuality.UNKNOWN,"current device motion unavailable") })
    }

    private fun onSemanticEvidence(evidence: GroundEvidence, sceneCurrent: Boolean, motion: MotionEvidence) {
        val key=evidence.key; val now=SystemClock.elapsedRealtime()
        if(!running || generation!=key.generation || (videoMode && (videoEpoch!=key.epoch || !videoPlaying))) return
        val rapid=motion.quality==MotionQuality.RAPID || imuMonitor?.currentDeviceMotion()?.quality==MotionQuality.RAPID || lastRapidAtMs>=key.capturedAtMs
        if(!sceneCurrent || rapid) environmentSpeechDeduplicator.reset()
        environmentEngine.observe(evidence,now,sceneCurrent,rapid)
        diagnostics.record("semantic",JSONObject().put("source_frame_key","${key.generation}:${key.epoch}:${key.frameIndex}").put("source_ms",key.timestampMs).put("source_age_ms",now-key.capturedAtMs).put("model_sha256",evidence.modelSha).put("model_input_size",evidence.inputSize).put("latency_ms",evidence.latencyMs).put("motion_quality",motion.quality.name).put("synchronized",motion.synchronized).put("scene_current",sceneCurrent).put("class_counts",JSONObject(evidence.cells.groupingBy{it.classId.toString()}.eachCount())).put("structures",JSONArray(environmentEngine.supports.map {environmentSupportJson(it)})).put("silence_reason",environmentEngine.silenceReason))
        Log.i("SG_ENV", "observed frame=${key.frameIndex} sourceMs=${key.timestampMs} ageMs=${now-key.capturedAtMs} sceneCurrent=$sceneCurrent rapid=$rapid classes=${evidence.cells.groupingBy{it.classId}.eachCount()} support=${environmentEngine.supports.map{environmentSupportJson(it)}} status=${environmentEngine.status} approach=false")
        if(!sceneCurrent || rapid) speech.cancelSemanticExcept(emptySet(),"semantic_scene_changed")
        else speech.cancelSemanticExcept(environmentEngine.validIds(now),if(now-key.capturedAtMs>environmentEngine.config.maxSourceAgeMs) "semantic_expired" else "semantic_absent")
        serviceEnvironment()
    }
    private fun serviceEnvironment() {
        val now=SystemClock.elapsedRealtime()
        if(!spatialEnabled) return
        if(imuMonitor?.currentDeviceMotion()?.quality==MotionQuality.RAPID) {
            lastRapidAtMs=now
            environmentEngine.invalidate(now,"墙面：设备快速转动，等待新鲜证据")
            environmentSpeechDeduplicator.reset()
            speech.cancelSemanticExcept(emptySet(),"semantic_rapid_motion")
        }
        val notices=environmentEngine.pending(now)
        environmentStatus=environmentEngine.status
        speech.cancelSemanticExcept(environmentEngine.validIds(now),"semantic_expired")
        if(!speech.isReady || muted || (videoMode && !videoPlaying) || pendingSpeech.isNotEmpty()) return
        for(notice in notices) {
            if(environmentSpeechDeduplicator.alreadyHeard(generation,if(videoMode) videoEpoch else 0L,notice,now)) {
                environmentEngine.acknowledgedByDetection(notice.id)
                diagnostics.record("semantic_dedup",JSONObject().put("event_id",notice.id).put("reason","overlapping_compatible_yolo_actual_onStart"))
                continue
            }
            if(notice.id in semanticPending || !speech.canQueueOrdinaryAlert) continue
            semanticPending.add(notice.id)
            Log.i("SG_ENV", "proposed event=${notice.id} frame=${notice.frameIndex} sourceMs=${notice.sourceMs} deadline=${notice.validUntilMs} text=${notice.text} riskScore=unchanged")
            if(!speech.offer(notice.text,false,notice.id,notice.validUntilMs,notice.direction,requireExactSpeech=false)) semanticPending.remove(notice.id)
        }
    }

    private fun describeEnvironment() {
        overviewSpeechNote=""
        if(imuMonitor?.currentDeviceMotion()?.quality==MotionQuality.RAPID) { spatialSnapshot=null; environmentOverview="当前环境信息不足。"; spatialStatus="UNKNOWN：设备快速转动"; return }
        val snapshot = spatialSnapshot
        val exactPaused = snapshot != null && videoMode && !videoPlaying && frame?.vision?.records?.frameIndex == snapshot.key.frameIndex
        val current = snapshot != null && (exactPaused || SystemClock.elapsedRealtime() - snapshot.key.capturedAtMs <= SpatialSessionAnalyzer.RESULT_TTL_MS)
        val primary = frame?.takeIf { ResultFreshness.isCurrent(it.receivedAtMs,SystemClock.elapsedRealtime()) }
        environmentOverview = SemanticEvidence.summarize(if(current) snapshot?.semantics?.cells ?: emptyList() else emptyList(), if(current) snapshot?.detections ?: emptyList() else primary?.vision?.raw ?: emptyList(), SemanticSourceQuality(current = current || primary != null, readable = current || primary != null)).overview
        Log.i("SG_OVERVIEW", "sourceFrame=${snapshot?.key?.frameIndex} current=$current text=$environmentOverview")
        if(running && !muted && environmentOverview!="当前环境信息不足。") {
            if(speech.canSpeakDynamic) speech.offer(environmentOverview,false,"overview-$generation-${SystemClock.elapsedRealtime()}",SystemClock.elapsedRealtime()+2500,requireExactSpeech=true)
            else { overviewSpeechNote="缺少离线动态中文语音，当前描述无法朗读。"; speechStatus=overviewSpeechNote }
        }
        if(muted) overviewSpeechNote="提醒已静音，仅显示描述文本。"
    }

    @Composable private fun SpatialPanel() {
        AlertDialog(onDismissRequest = { spatialPanel = false }, title = { Text("开发者工具 · 研究证据") }, confirmButton = { TextButton({ spatialPanel = false }) { Text("返回画面") } },
            text = { Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Row(verticalAlignment=Alignment.CenterVertically) { Text("打开应用自动感知",Modifier.weight(1f),fontSize=12.sp); Switch(autoStart,{autoStart=it; autoPolicy.enabled=it; getSharedPreferences("perception",MODE_PRIVATE).edit().putBoolean("auto_start",it).apply()},Modifier.semantics{contentDescription="前台自动启动感知"}) }
                Row(verticalAlignment = Alignment.CenterVertically) { Text("Depth Anything V2 Small", Modifier.weight(1f), fontSize = 12.sp); Switch(depthEnabled, { depthEnabled=it; startSpatialSession() }, Modifier.semantics { contentDescription = "启用相对深度" }) }
                Row(verticalAlignment = Alignment.CenterVertically) { Text("米制墙距研究 · 默认关闭",Modifier.weight(1f),fontSize=12.sp); Switch(metricResearchEnabled,{metricResearchEnabled=it;metricResearchStatus=if(it) "等待同帧研究证据，自动语音关闭" else "米制墙距研究关闭";startSpatialSession()},Modifier.semantics {contentDescription="启用米制墙距研究，不产生语音或导航建议"}) }
                Text(metricResearchStatus,fontSize=11.sp)
                Text("米制数据仅用于研究；P10/P90 是区域分布，不是准确率区间。局部走廊候选仍在 PC 验证。",fontSize=11.sp)
                Row(verticalAlignment = Alignment.CenterVertically) { Text("SegFormer ADE20K", Modifier.weight(1f), fontSize = 12.sp); Switch(spatialEnabled, { spatialEnabled=it; startSpatialSession() }, Modifier.semantics { contentDescription = "启用环境语义" }) }
                Row(verticalAlignment = Alignment.CenterVertically) { Text("15×15 Spatial Grid", Modifier.weight(1f), fontSize=12.sp); Switch(gridEnabled, { gridEnabled=it; startSpatialSession() }, Modifier.semantics { contentDescription="启用空间网格" }) }
                Row(verticalAlignment = Alignment.CenterVertically) { Text("IMU 运动质量", Modifier.weight(1f), fontSize=12.sp); Switch(imuEnabled, { imuEnabled=it; startSpatialSession() }, Modifier.semantics { contentDescription="启用运动传感器" }) }
                for (view in listOf("原始画面", "深度热图", "深度叠层", "环境语义", "Spatial Grid")) FilterChip(spaceView==view, { spaceView=view }, label={Text(view)})
                Text("语义图层", fontSize=12.sp)
                for (layer in listOf("Ground", "Wall", "Door", "Tree / Vegetation", "Static Structures", "All Semantics")) FilterChip(semanticLayer==layer, { semanticLayer=layer; spaceView="环境语义" }, label={Text(layer,fontSize=11.sp)})
                Text("叠层透明度 ${(layerAlpha*100).toInt()}%", fontSize=12.sp)
                Slider(layerAlpha, { layerAlpha=it }, valueRange=.1f.. .75f)
                Text("画面使用模型实际源帧快照；超期显示 UNKNOWN。暖色相对较近，冷色较远；不是米制距离。语义不是安全或碰撞判断。",fontSize=11.sp)
                Text(spatialStatus,fontSize=11.sp)
                Text(wallShadowStatus,fontSize=12.sp)
                Text("墙面接近：仅研究记录，自动播报关闭")
                Text("单级台阶与路缘：模型识别能力尚未验证，自动提醒关闭")
                Text(imuMonitor?.diagnostic() ?: "IMU UNKNOWN：未采集或文件时钟无法同步",fontSize=11.sp)
                spatialSnapshot?.let { snapshot ->
                    Text("源帧 ${snapshot.key.generation}:${snapshot.key.epoch}:${snapshot.key.frameIndex} · ${snapshot.key.width}×${snapshot.key.height} · 年龄 ${SystemClock.elapsedRealtime()-snapshot.key.capturedAtMs}ms · 时钟 ${snapshot.key.timestampClock}",fontSize=11.sp)
                    snapshot.detections.take(3).forEach { detection -> snapshot.depth?.region(detection)?.let { stats -> Text("${Labels.detectionNames.getOrElse(detection.classId){"目标"}}：${stats.relation} · 中位 ${stats.median?.let { "%.2f".format(it) } ?: "UNKNOWN"} · 有效 ${(stats.validFraction*100).toInt()}% · 框内可能混入背景",fontSize=10.sp) } }
                }
                Text("运行：$status · $inputStatus · %.1f FPS · ${latencyMs}ms".format(fps))
                Text("语音：$speechStatus")
                Text("最近事件：$lastEvent")
                Text("环境：$environmentStatus")
                OutlinedButton(::exportDiagnostics,Modifier.fillMaxWidth().heightIn(min=52.dp)) {Text("导出诊断记录")}
                if(videoMode) {
                    Text(videoName)
                    Button(::togglePlayback) {Text(if(videoPlaying) "暂停视频" else "播放视频")}
                    Text("${timeLabel(videoPosition)} / ${timeLabel(videoDuration)}")
                    Slider(videoPosition.toFloat(),{if(running) seekVideo(it.toLong()) else videoPosition=it.toLong()},valueRange=0f..videoDuration.coerceAtLeast(1).toFloat(),modifier=Modifier.semantics{contentDescription="视频分析时间轴"})
                }
                Button({ describeEnvironment() }) { Text("描述周围环境") }
                Text(environmentOverview,fontSize=12.sp)
            } })
    }

    @Composable private fun SettingsPanel() {
        AlertDialog(onDismissRequest={settingsPanel=false},title={Text("设置")},confirmButton={TextButton({settingsPanel=false}) {Text("返回")}},
            text={Column(Modifier.verticalScroll(rememberScrollState()),verticalArrangement=Arrangement.spacedBy(10.dp)) {
                Row(verticalAlignment=Alignment.CenterVertically) { Text("打开应用自动感知",Modifier.weight(1f)); Switch(autoStart,{autoStart=it;autoPolicy.enabled=it;getSharedPreferences("perception",MODE_PRIVATE).edit().putBoolean("auto_start",it).apply()},Modifier.semantics{contentDescription="前台自动启动感知"}) }
                Text("摄像头权限："+if(checkSelfPermission(Manifest.permission.CAMERA)==PackageManager.PERMISSION_GRANTED) "已授权" else "未授权")
                OutlinedButton({startActivity(Intent(android.provider.Settings.ACTION_APPLICATION_DETAILS_SETTINGS,Uri.parse("package:$packageName")))},Modifier.fillMaxWidth().heightIn(min=52.dp)) {Text("管理摄像头权限")}
                OutlinedButton({autoPolicy.sourceChanged(true);videoPicker.launch(arrayOf("video/*"));settingsPanel=false},Modifier.fillMaxWidth().heightIn(min=52.dp)) {Text("分析本地视频")}
                if(videoMode) OutlinedButton({stopPerception(false);videoMode=false;autoPolicy.sourceChanged(false);settingsPanel=false;requestStart()},Modifier.fillMaxWidth().heightIn(min=52.dp)) {Text("返回实时相机")}
                OutlinedButton({settingsPanel=false;spatialPanel=true},Modifier.fillMaxWidth().heightIn(min=52.dp)) {Text("开发者工具")}
                Row(verticalAlignment=Alignment.CenterVertically) {
                    Text("使用旧版目标模型",Modifier.weight(1f))
                    Switch(baselineDetector,{ baselineDetector=it
                        getSharedPreferences("perception",MODE_PRIVATE).edit().putBoolean("baseline_detector",it).apply()
                        if(running) {stopPerception(false);startPerception()}
                    },Modifier.semantics {contentDescription="切换旧版 YOLO26n；关闭使用 YOLOE"})
                }
                Text("此版本使用 CameraX 与离线 YOLOE；不提供距离或通行方向语音。",style=MaterialTheme.typography.bodySmall)
                Text("图像在本地处理。环境提示补充盲杖与导盲犬；未提醒不代表前方没有障碍。单级台阶、路缘及数值距离自动播报尚未启用。",style=MaterialTheme.typography.bodyMedium)
            }})
    }
    private fun exportDiagnostics() { exportDocument.launch("SightGuard-0.7-diagnostics.json") }
    private fun perceptionState(): String = when {
        !running -> if(status.contains("权限") || status.contains("失败")) "感知暂时受限。" else "感知已停止。"
        frame==null && !interrupted -> "正在启动。"
        interrupted || frame?.vision?.risk?.frame?.perception?.status!="available" || ((spatialEnabled || depthEnabled) && spatialStatus.startsWith("UNKNOWN")) -> "感知暂时受限。"
        else -> "正在感知。"
    }
    private fun currentAlertText(): String {
        if(!running) return if(status.contains("权限") || status.contains("失败")) "当前感知受限。" else "感知已停止。"
        val now=SystemClock.elapsedRealtime()
        val current=frame?.takeIf { ResultFreshness.isCurrent(it.receivedAtMs,now) }
        val risk=current?.vision?.risk?.frame?.events?.filter { it.observedNow && it.state==EventState.ACTIVE && it.assessment?.level in setOf(org.sightguard.app.core.RiskLevel.MEDIUM,org.sightguard.app.core.RiskLevel.HIGH) }?.maxByOrNull { it.assessment?.score ?: 0.0 }
        val environment=environmentEngine.currentNotices(now).take(2).map { it.text }
        val texts=mutableListOf<String>()
        if(risk!=null) texts.add("检测到${Labels.detectionNames.getOrElse(risk.classId){"目标"}}，请注意。")
        texts.addAll(environment)
        return texts.joinToString("\n").ifBlank { if(current==null || interrupted) "当前感知受限。" else "暂无已确认的提醒。" }
    }
    @Composable private fun Screen() {
        if(permissionExplanation) AlertDialog(onDismissRequest={permissionExplanation=false;autoPolicy.manualStop()},title={Text("摄像头权限用途")},
            text={Text("SightGuard 使用后置摄像头在设备本地识别环境，不上传画面。授权后前台自动感知；可随时停止、静音或在设置中修改权限。")},
            confirmButton={TextButton({permissionExplanation=false;permission.launch(Manifest.permission.CAMERA)}) {Text("继续授权")}},
            dismissButton={TextButton({permissionExplanation=false;autoPolicy.manualStop()}) {Text("暂不开启")}})
        if(settingsPanel) SettingsPanel()
        if(spatialPanel) SpatialPanel()
        // Reading displayNow subscribes to freshness changes; no automatic accessibility announcement.
        val now=displayNow
        val silence=when {
            muted -> "用户已静音。"
            !running -> status
            spatialStatus.startsWith("UNKNOWN") -> "当前模型运行受限。"
            now-lastValidAtMs>3500 && lastValidAtMs>0 -> "感知结果已过期。"
            else -> environmentEngine.silenceReason
        }
        AccessibleHome(perceptionState(),currentAlertText(),silence,lastSpeech,
            environmentOverview.takeIf { it!="当前环境信息不足。" }?.let {it+if(overviewSpeechNote.isBlank()) "" else "\n$overviewSpeechNote"},running,muted,
            ::describeEnvironment,{muted=!muted;speech.setMuted(muted);getSharedPreferences("perception",MODE_PRIVATE).edit().putBoolean("muted",muted).apply()},
            {if(running) {autoPolicy.manualStop();stopPerception()} else requestStart()}, {settingsPanel=true}) { PreviewContent() }
    }
    @Composable private fun PreviewContent() {
        val snapshot=spatialSnapshot
        val available=when(spaceView) {
            "深度热图","深度叠层" -> snapshot?.depth?.quality=="VALID" && snapshot.heatmap!=null
            "环境语义" -> snapshot?.semantics!=null
            "Spatial Grid" -> snapshot?.grid!=null
            else -> false
        }
        Box(Modifier.fillMaxSize()) {
            if(videoMode) videoBitmap?.let {Image(it.asImageBitmap(),null,Modifier.fillMaxSize(),contentScale=ContentScale.Fit)}
            else AndroidView(factory={previewView},modifier=Modifier.fillMaxSize())
            if(spaceView!="原始画面" && available && snapshot!=null) {
                Image(snapshot.bitmap.asImageBitmap(),null,Modifier.fillMaxSize(),contentScale=ContentScale.Fit)
                if(spaceView.startsWith("深度")) snapshot.heatmap?.let {Image(it.asImageBitmap(),null,Modifier.fillMaxSize(),contentScale=ContentScale.Fit,alpha=if(spaceView=="深度热图") 1f else layerAlpha)}
                if(spaceView=="环境语义") SemanticOverlay(snapshot.semantics,snapshot.key.width,snapshot.key.height)
                if(spaceView=="Spatial Grid") GridOverlay(snapshot.grid,snapshot.key.width,snapshot.key.height)
                if(boxesEnabled) BoxesOverlay(snapshot.detections,snapshot.key.width,snapshot.key.height)
            } else if(boxesEnabled) DetectionOverlay(frame)
        }
    }
    @Composable private fun DetectionOverlay(result: FrameResult?) {
        if (result != null) BoxesOverlay(result.vision.raw, result.width, result.height,null)
    }
    @Composable private fun BoxesOverlay(detections: List<Detection>, sourceWidth: Int, sourceHeight: Int, arMapping: ArPreviewMapping?=null) {
        Canvas(Modifier.fillMaxSize().semantics { contentDescription = "${detections.size} 个检测目标" }) {
            val fit = fitImage(sourceWidth, sourceHeight, size.width, size.height)
            val width = sourceWidth * fit.scale; val height = sourceHeight * fit.scale
            val dx = fit.offsetX; val dy = fit.offsetY
            val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = android.graphics.Color.WHITE; textSize = 11.sp.toPx(); setShadowLayer(3f, 0f, 1f, android.graphics.Color.BLACK) }
            detections.forEach { d ->
                val mapped=arMapping?.point(d.left,d.top);val end=arMapping?.point(d.right,d.bottom)
                val x = mapped?.first?.times(size.width) ?: (dx + d.left * width); val y = mapped?.second?.times(size.height) ?: (dy + d.top * height)
                val right=end?.first?.times(size.width) ?: (dx+d.right*width);val bottom=end?.second?.times(size.height) ?: (dy+d.bottom*height)
                val leftLimit=(1-(arMapping?.widthFraction ?: 1f))*size.width/2
                val topLimit=(1-(arMapping?.heightFraction ?: 1f))*size.height/2
                val l=x.coerceIn(leftLimit,size.width-leftLimit);val t=y.coerceIn(topLimit,size.height-topLimit)
                val r=right.coerceIn(leftLimit,size.width-leftLimit);val b=bottom.coerceIn(topLimit,size.height-topLimit)
                if(r>l && b>t) {
                    drawRect(Color(0xFFFF4E62),Offset(l,t),Size(r-l,b-t),style=Stroke(2.dp.toPx()))
                    drawContext.canvas.nativeCanvas.drawText("${Labels.detectionNames[d.classId]}${if(d.classId==43) "（类别不确定）" else ""} ${(d.confidence*100).toInt()}%",l,(t-4.dp.toPx()).coerceAtLeast(topLimit+16.sp.toPx()),paint)
                }
            }
        }
    }
    @Composable private fun SemanticOverlay(evidence: GroundEvidence?, sourceWidth: Int, sourceHeight: Int) {
        Canvas(Modifier.fillMaxSize().semantics { contentDescription = "真实环境语义，不表示安全区域" }) {
            if (evidence == null) return@Canvas
            val fit = fitImage(sourceWidth,sourceHeight,size.width,size.height)
            val width=sourceWidth*fit.scale; val height=sourceHeight*fit.scale
            evidence.cells.forEach { cell ->
                val group=SemanticEvidence.group(cell.classId,(cell.top+cell.bottom)/2)
                val selected=when(semanticLayer) {
                    "Ground" -> cell.classId in SpatialDetector.GROUND_CLASSES
                    "Wall" -> cell.classId==0
                    "Door" -> cell.classId==14
                    "Tree / Vegetation" -> cell.classId in setOf(4,9,17)
                    "Static Structures" -> cell.classId in setOf(1,10,15,19,32,34,38,42,53,59,87,93,95,96,121)
                    else -> true
                }
                if (selected) {
                    val color=when(group) {
                        SemanticGroup.WALL -> Color(0xFFFF7097)
                        SemanticGroup.DOOR -> Color(0xFFB390FF)
                        SemanticGroup.TREE_UPPER,SemanticGroup.TREE_LOWER,SemanticGroup.PLANT,SemanticGroup.GRASS -> Color(0xFF60BFA5)
                        SemanticGroup.FLOOR,SemanticGroup.ROAD,SemanticGroup.SIDEWALK -> Color(0xFFE8BD62)
                        else -> Color(0xFF79A8DB)
                    }.copy(alpha=layerAlpha)
                    drawRect(color,Offset(fit.offsetX+cell.left*width,fit.offsetY+cell.top*height),Size((cell.right-cell.left)*width,(cell.bottom-cell.top)*height))
                }
            }
        }
    }
    @Composable private fun GridOverlay(grid: SpatialGridEvidence?, sourceWidth:Int, sourceHeight:Int) {
        if(grid==null) return
        Box(Modifier.fillMaxSize()) {
            Canvas(Modifier.fillMaxSize().semantics { contentDescription="15乘15图像证据网格，不表示安全路径" }) {
                val fit=fitImage(sourceWidth,sourceHeight,size.width,size.height)
                val width=sourceWidth*fit.scale; val height=sourceHeight*fit.scale
                grid.cells.forEach { cell ->
                    val color=when { cell.quality=="DEGRADED"->Color.Gray; cell.state=="CONFLICT"->Color(0xFFFF66AA); cell.state=="OBJECT"->Color(0xFFFF6655); cell.state=="GROUND"->Color(0xFFFFCC66); cell.state=="WALL"->Color(0xFFAA88EE); cell.state=="VEGETATION"->Color(0xFF77BBBB); else->Color.Gray }
                    val origin=Offset(fit.offsetX+cell.left*width,fit.offsetY+cell.top*height); val extent=Size((cell.right-cell.left)*width,(cell.bottom-cell.top)*height)
                    drawRect(color.copy(alpha=layerAlpha),origin,extent)
                    drawRect(Color.White.copy(alpha=.25f),origin,extent,style=Stroke(1f))
                }
            }
            Text("${grid.state} · 运动 ${grid.motion.quality}\n黄=地面 紫=墙 红=检测框/家具语义 粉=冲突 青=植被 灰=UNKNOWN/降级\n同帧研究证据，非安全通行网格",Modifier.align(Alignment.TopStart).padding(top=30.dp,start=8.dp).background(Color(0xCC090E17)).padding(5.dp),fontSize=9.sp,color=Color.White)
        }
    }
    internal fun sessionDiagnostics(): JSONObject = JSONObject().put("arActive",false).put("cameraBackend","CameraX").put("detectorModelSha",stage?.modelSha).put("running",running).put("generation",generation).put("modelReady",stage!=null).put("frameIndex",frame?.vision?.records?.frameIndex ?: -1).put("source",if(videoMode) "video" else "camera").put("explicitStop",autoPolicy.explicitlyStopped).put("releasePending",primaryReleasePending).put("spatialReleasePending",spatialReleasePending).put("semanticEnabled",spatialEnabled).put("debugThermalOverride",ignoreAuxThermalForTest).put("muted",muted).put("settingsPanel",settingsPanel).put("developerPanel",spatialPanel).put("currentAlert",currentAlertText()).put("environmentStatus",environmentStatus).put("fps",fps).put("latencyMs",latencyMs).put("speechStatus",speechStatus).put("lastSpeech",lastSpeech)
    private fun timeLabel(ms: Long) = "%02d:%02d".format(ms / 60000, ms / 1000 % 60)
    private fun recordJson(result: VisionResult): JSONObject = JSONObject()
        .put("android_relevance_policy", "android-relevance-1.0")
        .put("risk_relevance", JSONArray().apply { result.relevance.forEach { d -> put(JSONObject().put("detection_index",d.detectionIndex).put("track_id",d.trackId).put("class_id",d.classId).put("category",d.category.name).put("eligible",d.eligible).put("class_uncertain",d.classUncertain).put("reasons",JSONArray(d.reasons))) } })
        .put("frame_index", result.records.frameIndex).put("timestamp_ms", result.records.timestampMs)
        .put("frame_width", result.records.frameWidth).put("frame_height", result.records.frameHeight).put("timestamp_source", result.records.timestampSource).put("status", result.records.status)
        .put("detections", JSONArray().apply { result.records.detections.forEach { d -> put(JSONObject().put("track_id", d.trackId).put("class_id", d.classId).put("class_name", d.className).put("confidence", d.confidence).put("bbox_xyxy", JSONArray(d.bboxXyxy))) } })
}
