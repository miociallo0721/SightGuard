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
    private var running by mutableStateOf(false)
    private var status by mutableStateOf("待启动")
    private var inputStatus by mutableStateOf("摄像头未启动")
    private var speechStatus by mutableStateOf("开始后自动启用离线中文提醒")
    private var lastSpeech by mutableStateOf("尚未播放提醒")
    private var lastEvent by mutableStateOf("尚无已确认事件")
    private var muted by mutableStateOf(false)
    private var boxesEnabled by mutableStateOf(true)
    private var riskEnabled by mutableStateOf(true)
    private var spatialEnabled by mutableStateOf(false)
    private var spatialStatus by mutableStateOf("空间语义关闭")
    private var ground by mutableStateOf<GroundEvidence?>(null)
    @Volatile private var spatialAnalyzer: SpatialAnalyzer? = null
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
        if (granted) startPerception() else status = "摄像头权限被拒绝，请在系统设置中授权"
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
            val evidence = ground
            if (evidence != null && !(videoMode && !videoPlaying && frame?.vision?.records?.frameIndex == evidence.key.frameIndex) && SystemClock.elapsedRealtime() - evidence.key.capturedAtMs > SpatialDetector.RESULT_TTL_MS) ground = null
            if (running && !videoMode) {
                val now = SystemClock.elapsedRealtime()
                if (frame?.let { !ResultFreshness.isCurrent(it.receivedAtMs, now) } == true) { frame = null; fps = 0f }
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
        previewView = PreviewView(this).apply { implementationMode = PreviewView.ImplementationMode.COMPATIBLE; scaleType = PreviewView.ScaleType.FIT_CENTER }
        speech = SpeechController(this, { speechStatus = it }, { lastSpeech = it }, ::onPlayback)
        setContent { MaterialTheme(colorScheme = darkColorScheme(primary = Color(0xFF74DFCB), background = Color(0xFF090E17), surface = Color(0xFF17212F))) { Screen() } }
        handler.post(watchdog)
    }
    private fun requestStart() {
        if (videoMode && videoUri == null) { videoPicker.launch(arrayOf("video/*")); return }
        if (videoMode || checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startPerception()
        else permission.launch(Manifest.permission.CAMERA)
    }
    private fun startPerception() {
        if (running || isFinishing || isDestroyed || !lifecycle.currentState.isAtLeast(androidx.lifecycle.Lifecycle.State.STARTED)) return
        val token = ++generation
        running = true; active = true; frame = null; videoBitmap = null; fps = 0f; latencyMs = 0; lastValidAtMs = 0
        interrupted = false; pendingSpeech.clear(); startedAtMs = SystemClock.elapsedRealtime(); videoRateStart = 0; videoRateFrames = 0
        status = "正在加载本地 YOLO26n…"; inputStatus = if (videoMode) "读取本地视频" else "等待后置摄像头"
        lastEvent = "尚无已确认事件"; videoPlaying = videoMode; videoBusy = false; processedVideoEpoch = -1; videoEpoch++
        speech.setMuted(muted); speech.start()
        if (spatialEnabled) startSpatial()
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
                    if (isVideo) {
                        videoName = ownedVideo!!.name; videoDuration = ownedVideo!!.durationMs
                        videoPosition = videoPosition.coerceIn(0, (videoDuration - 1).coerceAtLeast(0))
                        nextVideoPosition = videoPosition
                        inputStatus = "本地解码 · 请求位置估计时间 · CPU"; pumpVideo(token)
                    } else bindCamera(token, executor, ownedStage!!)
                }
            } catch (error: Exception) { Log.e("SG_MODEL", "session load failed", error); handler.post { if (generation == token) fail("输入或模型加载失败：${error.message}") } }
        }
        Log.i("SG_SESSION", "start generation=$token source=${if (isVideo) "video" else "CameraX"}")
    }
    @Suppress("DEPRECATION")
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
                        if (firstTimestampNs == null) firstTimestampNs = sensor
                        val timestamp = (sensor - firstTimestampNs!!) / 1_000_000
                        if (timestamp <= previousTimestampMs) return@setAnalyzer
                        previousTimestampMs = timestamp
                        val bitmap = uprightBitmap(image); val width = bitmap.width; val height = bitmap.height
                        val result = try {
                            vision.process(bitmap, timestamp, "monotonic").also { v ->
                                spatialAnalyzer?.offer(bitmap, SpatialFrameKey(token, 0, v.records.frameIndex, v.records.timestampMs, width, height, receivedAt))
                            }
                        } finally { bitmap.recycle() }
                        val completedAt = SystemClock.elapsedRealtime(); val latency = completedAt - receivedAt
                        if (!ResultFreshness.isCurrent(receivedAt, completedAt)) { Log.w("SG_FRAME", "discard stale latencyMs=$latency"); return@setAnalyzer }
                        if (trace != null && completed < 200) trace.appendText(recordJson(result).toString() + "\n")
                        completed++; rateFrames++
                        val rate = rateFrames * 1000f / (completedAt - rateStart).coerceAtLeast(1)
                        if (completedAt - rateStart >= 3000) { rateStart = completedAt; rateFrames = 0 }
                        if (completed % 10 == 1) Log.i("SG_FRAME", "frame=$completed sensorMs=$timestamp size=${width}x$height objects=${result.raw.size} latencyMs=$latency fps=$rate activeEvents=${result.risk.frame.events.count { it.state == EventState.ACTIVE }}")
                        handler.post {
                            if (!running || generation != token || !ResultFreshness.isCurrent(receivedAt, SystemClock.elapsedRealtime())) return@post
                            publish(result, width, height, receivedAt, latency, rate, vision, token, 0)
                        }
                    } catch (error: Exception) { Log.e("SG_FRAME", "inference failed", error); handler.post { if (running && generation == token) fail("视觉感知失败：${error.message}") } }
                    finally { image.close() }
                }
                val viewport = previewView.viewPort ?: ViewPort.Builder(Rational(3, 4), rotation).setScaleType(ViewPort.FIT).build()
                val camera = cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, UseCaseGroup.Builder().setViewPort(viewport).addUseCase(preview).addUseCase(analyzer).build())
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
                spatialAnalyzer?.offer(video.bitmap, SpatialFrameKey(token, epoch, result.records.frameIndex, result.records.timestampMs, video.bitmap.width, video.bitmap.height, SystemClock.elapsedRealtime()))
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
        frame = FrameResult(result, width, height, receivedAt); lastValidAtMs = SystemClock.elapsedRealtime(); latencyMs = latency; fps = rate
        status = if (videoMode && !videoPlaying) "视频已暂停" else "感知运行中"
        if (interrupted) { speech.offer("感知已恢复", true, "recovered-$token", SystemClock.elapsedRealtime() + 2000); interrupted = false }
        if (!speech.isReady || muted || (videoMode && !videoPlaying)) return
        val now = SystemClock.elapsedRealtime()
        val waitingEvents = pendingSpeech.values.flatMap { (it.presentation.merged + it.presentation.alert).map { a -> a.eventId } }.toSet()
        val proposal = vision.presenter.choose(result.risk.copy(alerts = result.risk.alerts.filter { it.eventId !in waitingEvents }), now) ?: return
        if (!proposal.important && !speech.canQueueOrdinaryAlert) return
        if (pendingSpeech.size >= 2 && !proposal.important) return
        lastEvent = "${proposal.alert.eventId} · ${proposal.alert.alertType} · ${proposal.text}"
        Log.i("SG_ALERT", "raw event=${proposal.alert.eventId} type=${proposal.alert.alertType} sourceMs=${proposal.alert.timestampMs} score=${proposal.alert.riskScore} text=${proposal.text}")
        val key = "$token:$epoch:${proposal.alert.eventId}:${proposal.alert.alertType}:${++speechSequence}"
        pendingSpeech[key] = PendingSpeech(token, epoch, vision, proposal, result, key)
        if (!speech.offer(proposal.text, proposal.important, key, vision.presenter.deadline(now))) pendingSpeech.remove(key)
    }
    private fun onPlayback(event: SpeechPlaybackEvent) {
        val pending = pendingSpeech[event.eventId] ?: return
        when (event.phase) {
            SpeechPlaybackPhase.ON_START -> {
                pendingSpeech.remove(event.eventId)
                if (running && generation == pending.token && (!videoMode || videoEpoch == pending.epoch)) {
                    pending.stage.presenter.accepted(pending.presentation, pending.result.risk, event.elapsedRealtimeMs)
                    worker?.execute { if (active && generation == pending.token) pending.stage.accept(pending.presentation) }
                }
            }
            SpeechPlaybackPhase.SUPPRESSED, SpeechPlaybackPhase.CANCELLED -> pendingSpeech.remove(event.eventId)
            else -> Unit
        }
    }
    private fun seekVideo(position: Long) {
        ground = null
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
        val ready = speech.isReady
        stopPerception(false, !ready); status = message
        if (!ready) return
        speech.offer(if (message.startsWith("摄像头")) "摄像头连接失败" else "视觉感知暂时中断", true, "failure-$generation", SystemClock.elapsedRealtime() + 2500)
        handler.postDelayed({ if (!running) speech.stop(false) }, 3000)
    }
    private fun stopPerception(announce: Boolean = true, stopSpeech: Boolean = true) {
        if (!running) { if (stopSpeech) speech.stop(false); return }
        running = false; active = false; generation++; pendingSpeech.clear(); videoPlaying = false; videoEpoch++
        spatialAnalyzer?.close(); spatialAnalyzer = null; ground = null; spatialStatus = "空间语义已释放"
        cameraState?.removeObservers(this); cameraState = null; analysis?.clearAnalyzer(); analysis = null; provider?.unbindAll(); provider = null
        val release = releaseSession; releaseSession = null
        worker?.let { executor -> executor.execute { try { release?.invoke() } finally { Log.i("SG_SESSION", "model and input released") } }; executor.shutdown() }
        worker = null; stage = null; videoSource = null; videoBusy = false
        frame = null; videoBitmap = null; fps = 0f; latencyMs = 0; lastValidAtMs = 0
        inputStatus = if (videoMode) "文件解码器已释放" else "摄像头已释放"; status = "感知已停止"
        if (stopSpeech) speech.stop(announce)
        Log.i("SG_SESSION", "stop input unbound generation=$generation")
    }
    override fun onStop() { stopPerception(false); super.onStop() }
    override fun onDestroy() { stopPerception(false); handler.removeCallbacksAndMessages(null); speech.close(); super.onDestroy() }

    private fun startSpatial() {
        spatialAnalyzer?.close(); ground = null; spatialStatus = "正在加载地面语义模型…"
        val token = generation
        spatialAnalyzer = SpatialAnalyzer(this, { result ->
            if (running && generation == result.key.generation && (!videoMode || videoEpoch == result.key.epoch)) {
                val matches = !videoMode || frame?.vision?.records?.frameIndex == result.key.frameIndex
                if (matches && (videoMode && !videoPlaying || SystemClock.elapsedRealtime() - result.key.capturedAtMs <= SpatialDetector.RESULT_TTL_MS)) ground = result
                spatialStatus = "${result.state} · ${result.latencyMs}ms"
            }
        }, { state ->
            if (running && generation == token) {
                spatialStatus = state
                val bitmap = videoBitmap; val records = frame?.vision?.records
                if (videoMode && !videoPlaying && bitmap != null && records != null) spatialAnalyzer?.offer(bitmap, SpatialFrameKey(token, videoEpoch, records.frameIndex, records.timestampMs, bitmap.width, bitmap.height, SystemClock.elapsedRealtime()), force = true)
            }
        })
    }

    @Composable private fun Screen() {
        Surface(Modifier.fillMaxSize()) {
            Column(Modifier.fillMaxSize().safeDrawingPadding().padding(14.dp), verticalArrangement = Arrangement.spacedBy(7.dp)) {
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) { Text("SightGuard", style = MaterialTheme.typography.headlineSmall, color = MaterialTheme.colorScheme.primary); Text("视界 · 本地感知", fontSize = 12.sp) }
                    Text("研究原型", color = Color(0xFFFFCB85), fontSize = 12.sp)
                }
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    FilterChip(selected = !videoMode, onClick = { if (videoMode) { stopPerception(false); videoMode = false; status = "待启动" } }, label = { Text("实时相机") })
                    FilterChip(selected = videoMode, onClick = { if (!videoMode) { stopPerception(false); videoMode = true; status = "选择本地视频" } }, label = { Text("文件分析") })
                    if (videoMode) TextButton(onClick = { videoPicker.launch(arrayOf("video/*")) }) { Text("选择视频") }
                }
                Text(status, style = MaterialTheme.typography.titleSmall)
                Box(Modifier.weight(1f).fillMaxWidth().clip(RoundedCornerShape(16.dp)).background(Color.Black)) {
                    if (videoMode) { videoBitmap?.let { Image(it.asImageBitmap(), "当前实际分析的视频帧", Modifier.fillMaxSize(), contentScale = ContentScale.Fit) } }
                    else AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize().semantics { contentDescription = "后置摄像头实时预览" })
                    if (spatialEnabled) SemanticOverlay(ground, frame)
                    if (boxesEnabled) DetectionOverlay(frame)
                    if (frame == null && (!running || videoMode)) Text(if (videoMode) videoName else "点击开始感知", Modifier.align(Alignment.Center).padding(20.dp), color = Color.LightGray)
                    Column(Modifier.align(Alignment.BottomStart).padding(10.dp).background(Color(0xD0090E17)).padding(6.dp)) {
                        Text("YOLO26n · ONNX CPU · 离线", color = Color.White, fontSize = 11.sp)
                        if (riskEnabled) {
                            val highest = frame?.vision?.risk?.frame?.events?.filter { it.observedNow && it.state == EventState.ACTIVE }?.maxByOrNull { it.assessment!!.score }
                            Text(if (highest == null) "尚无已确认风险事件" else "风险 ${highest.assessment!!.level} · %.2f · %s".format(highest.assessment!!.score, highest.eventId), color = Color(0xFFFFA1A5), fontSize = 12.sp)
                            Text("感知 ${frame?.vision?.risk?.frame?.perception?.status ?: "未运行"}", color = Color.LightGray, fontSize = 10.sp)
                        }
                    }
                }
                if (videoMode) {
                    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) { TextButton(::togglePlayback, enabled = videoUri != null) { Text(if (videoPlaying) "暂停" else "播放") }; Text("${timeLabel(videoPosition)} / ${timeLabel(videoDuration)}", fontSize = 12.sp, modifier = Modifier.weight(1f)) }
                    var slider by remember(videoDuration) { mutableFloatStateOf(videoPosition.toFloat()) }
                    var dragging by remember { mutableStateOf(false) }
                    LaunchedEffect(videoPosition) { if (!dragging) slider = videoPosition.toFloat() }
                    Slider(slider, { dragging = true; slider = it }, onValueChangeFinished = { dragging = false; if (running) seekVideo(slider.toLong()) else videoPosition = slider.toLong() }, valueRange = 0f..videoDuration.coerceAtLeast(1).toFloat(), enabled = videoDuration > 0, modifier = Modifier.heightIn(min = 48.dp).semantics { contentDescription = "视频分析时间轴" })
                }
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) { Text("目标 ${frame?.vision?.raw?.size ?: 0}", fontSize = 12.sp); Text("%.1f FPS".format(fps), fontSize = 12.sp); Text("${latencyMs} ms", fontSize = 12.sp) }
                Text(inputStatus, color = Color.LightGray, fontSize = 10.sp, maxLines = 1)
                if (spatialEnabled) Text(spatialStatus, color = Color(0xFFFFCB85), fontSize = 10.sp, maxLines = 1)
                Card(Modifier.fillMaxWidth()) { Column(Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(3.dp)) { Text("语音：$lastSpeech", fontSize = 14.sp, maxLines = 1); Text(speechStatus, fontSize = 10.sp, color = MaterialTheme.colorScheme.primary, maxLines = 2); Text(lastEvent, fontSize = 10.sp, color = Color.LightGray, maxLines = 1) } }
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    FilterChip(boxesEnabled, { boxesEnabled = !boxesEnabled }, label = { Text("检测框", fontSize = 11.sp) }); Spacer(Modifier.width(6.dp))
                    FilterChip(riskEnabled, { riskEnabled = !riskEnabled }, label = { Text("风险", fontSize = 11.sp) }); Spacer(Modifier.weight(1f)); Text("静音", fontSize = 12.sp)
                    Switch(muted, { muted = it; speech.setMuted(it) }, Modifier.semantics { contentDescription = "静音语音提醒" })
                }
                FilterChip(spatialEnabled, {
                    spatialEnabled = !spatialEnabled
                    if (spatialEnabled && running) startSpatial()
                    else { spatialAnalyzer?.close(); spatialAnalyzer = null; ground = null; spatialStatus = "空间语义关闭" }
                }, label = { Text("实验地面语义 · 低频 · 非安全区域", fontSize = 10.sp) })
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    Button(::requestStart, Modifier.weight(1f).heightIn(min = 52.dp), enabled = !running) { Text(if (videoMode) "开始分析" else "开始感知") }
                    OutlinedButton({ stopPerception() }, Modifier.weight(1f).heightIn(min = 52.dp), enabled = running) { Text("停止感知") }
                }
                Text("图像启发式风险分数，非碰撞概率；辅助盲杖与导盲犬，不提供安全导航保证。", fontSize = 10.sp, color = Color(0xFFABB8C8))
            }
        }
    }
    @Composable private fun DetectionOverlay(result: FrameResult?) {
        Canvas(Modifier.fillMaxSize().semantics { contentDescription = "${result?.vision?.raw?.size ?: 0} 个检测目标" }) {
            if (result == null) return@Canvas
            val fit = fitImage(result.width, result.height, size.width, size.height)
            val width = result.width * fit.scale; val height = result.height * fit.scale
            val dx = fit.offsetX; val dy = fit.offsetY
            val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = android.graphics.Color.WHITE; textSize = 11.sp.toPx(); setShadowLayer(3f, 0f, 1f, android.graphics.Color.BLACK) }
            result.vision.raw.forEach { d ->
                val x = dx + d.left * width; val y = dy + d.top * height
                drawRect(Color(0xFFFF4E62), Offset(x, y), Size((d.right - d.left) * width, (d.bottom - d.top) * height), style = Stroke(2.dp.toPx()))
                drawContext.canvas.nativeCanvas.drawText("${Labels.coco80[d.classId]} ${(d.confidence * 100).toInt()}%", x, (y - 4.dp.toPx()).coerceAtLeast(16.sp.toPx()), paint)
            }
        }
    }
    @Composable private fun SemanticOverlay(evidence: GroundEvidence?, result: FrameResult?) {
        Canvas(Modifier.fillMaxSize().semantics { contentDescription = "模型地面语义覆盖层，不表示安全区域" }) {
            if (evidence == null || result == null || evidence.key.width != result.width || evidence.key.height != result.height) return@Canvas
            if (videoMode && evidence.key.frameIndex != result.vision.records.frameIndex) return@Canvas
            val fit = fitImage(result.width, result.height, size.width, size.height)
            val width = result.width * fit.scale; val height = result.height * fit.scale
            evidence.cells.forEach { cell ->
                drawRect(Color(0x50FFBA55), Offset(fit.offsetX + cell.left * width, fit.offsetY + cell.top * height), Size((cell.right - cell.left) * width, (cell.bottom - cell.top) * height))
            }
        }
    }
    private fun timeLabel(ms: Long) = "%02d:%02d".format(ms / 60000, ms / 1000 % 60)
    private fun recordJson(result: VisionResult): JSONObject = JSONObject().put("frame_index", result.records.frameIndex).put("timestamp_ms", result.records.timestampMs)
        .put("frame_width", result.records.frameWidth).put("frame_height", result.records.frameHeight).put("timestamp_source", result.records.timestampSource).put("status", result.records.status)
        .put("detections", JSONArray().apply { result.records.detections.forEach { d -> put(JSONObject().put("track_id", d.trackId).put("class_id", d.classId).put("class_name", d.className).put("confidence", d.confidence).put("bbox_xyxy", JSONArray(d.bboxXyxy))) } })
}
