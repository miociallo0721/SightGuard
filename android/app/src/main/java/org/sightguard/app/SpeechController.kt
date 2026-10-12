package org.sightguard.app

import android.content.Context
import android.media.MediaPlayer
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.speech.tts.Voice
import android.util.Log
import java.util.UUID

enum class SpeechPlaybackPhase(val logName: String) {
    REQUESTED("requested"),
    ACCEPTED("accepted"),
    ON_START("onStart"),
    COMPLETED("completed"),
    SUPPRESSED("suppressed"),
    CANCELLED("cancelled"),
}

data class SpeechPlaybackEvent(
    val phase: SpeechPlaybackPhase,
    val eventId: String?,
    /** Text requested by the caller, even when a fixed local clip says a shorter phrase. */
    val requestedText: String,
    /** Actual text on start/completion; null for events without playback. */
    val spokenText: String?,
    val important: Boolean,
    val elapsedRealtimeMs: Long,
    val validUntilElapsedMs: Long,
    val detail: String? = null,
    val resourceName: String? = null,
    /** Speech session token, so consumers can ignore callbacks from a previous session. */
    val sessionGeneration: Long = 0L,
    val direction: Direction? = null,
)

internal data class SpeechRequest(
    val text: String,
    val important: Boolean,
    val eventId: String?,
    val validUntilElapsedMs: Long,
    val direction: Direction? = null,
    val requireExactSpeech: Boolean = false,
)

internal data class SpeechQueueOffer(val accepted: Boolean, val replaced: SpeechRequest? = null)

internal object SpeechRequestRules {
    fun isExpired(request: SpeechRequest, nowElapsedMs: Long): Boolean =
        nowElapsedMs >= request.validUntilElapsedMs

    fun preempts(incoming: SpeechRequest, active: SpeechRequest): Boolean =
        priority(incoming)>priority(active)
    fun priority(request: SpeechRequest): Int = when {
        request.important -> 5
        request.eventId?.startsWith("semantic:")==true -> 3
        request.eventId?.startsWith("overview-")==true -> 2
        request.eventId?.startsWith("research:")==true -> 2
        request.eventId?.startsWith("status-")==true -> 1
        else -> 4
    }
}

/** A single expiring waiting slot; the active utterance is managed by SpeechController. */
internal class SpeechRequestQueue {
    private var pending: SpeechRequest? = null

    fun offer(request: SpeechRequest): SpeechQueueOffer {
        val current = pending
        if (current == null) {
            pending = request
            return SpeechQueueOffer(accepted = true)
        }
        if (SpeechRequestRules.priority(request)>SpeechRequestRules.priority(current)) {
            pending = request
            return SpeechQueueOffer(accepted = true, replaced = current)
        }
        return SpeechQueueOffer(accepted = false)
    }

    fun take(): SpeechRequest? = pending.also { pending = null }

    fun peek(): SpeechRequest? = pending

    fun discardExpired(nowElapsedMs: Long): SpeechRequest? {
        val request = pending ?: return null
        if (!SpeechRequestRules.isExpired(request, nowElapsedMs)) return null
        pending = null
        return request
    }

    fun clear(): SpeechRequest? = pending.also { pending = null }
}

internal data class SpeechPromptSelection(val resourceName: String, val spokenText: String, val usedGeneric: Boolean)

/** Exact catalog for fixed clips; unmatched requests select the existing generic clip explicitly. */
internal object SpeechPromptCatalog {
    private val directions = listOf("left" to "左侧", "center" to "前方", "right" to "右侧")
    private val categories = listOf(
        "pedestrian" to "行人",
        "bicycle" to "自行车",
        "car" to "汽车",
        "motorcycle" to "摩托车",
        "bus" to "公交车",
        "truck" to "卡车",
        "dog" to "狗",
        "chair" to "椅子",
        "traffic_light" to "红绿灯",
        "stop_sign" to "停止标志",
        "train" to "火车",
        "target" to "目标",
    )
    private val importantCategories = setOf(
        "pedestrian", "bicycle", "car", "motorcycle", "bus", "truck", "traffic_light", "stop_sign", "target",
    )
    private val multipleKinds = listOf(
        "multiple_pedestrians" to "有多名行人",
        "multiple_targets" to "有多个目标",
    )

    fun resolve(text: String): SpeechPromptSelection {
        val fixed = mapOf(
            "感知已开启" to "perception_started",
            "感知已停止" to "perception_stopped",
            "视觉感知暂时中断" to "perception_interrupted",
            "摄像头连接失败" to "camera_connection_failed",
            "感知暂时受限" to "perception_limited",
            "感知已恢复" to "perception_recovered",
            "摄像头已停止" to "camera_stopped",
            "前方检测到墙面。" to "environment_wall",
            "前方检测到楼梯。" to "environment_stairs",
            "前方检测到台阶。" to "environment_step",
            "前方检测到杆柱。" to "environment_pole",
            "前方墙面可能较近，请注意。" to "ar_wall_near",
            "相机可能正在靠近前方墙面，请注意。" to "ar_wall_approaching",
            "画面左侧可见地面较连续，通行条件尚未确认。" to "ar_corridor_left",
            "画面中央可见地面较连续，通行条件尚未确认。" to "ar_corridor_center",
            "画面右侧可见地面较连续，通行条件尚未确认。" to "ar_corridor_right",
        )
        fixed[text]?.let { return SpeechPromptSelection(it, text, usedGeneric = false) }
        for(kind in EnvironmentKind.entries) for((key,direction) in listOf("left" to Direction.LEFT,"center" to Direction.CENTER,"right" to Direction.RIGHT)) {
            if(text==environmentText(kind,direction)) return SpeechPromptSelection("environment_${kind.name.lowercase()}_$key",text,false)
        }

        for ((directionKey, directionText) in directions) {
            val elevatedText = "${directionText}目标风险升高，请注意"
            if (text == elevatedText) {
                return SpeechPromptSelection("alert_${directionKey}_risk_elevated", elevatedText, usedGeneric = false)
            }
            for ((categoryKey, categoryText) in categories) {
                val spokenText = "${directionText}有$categoryText"
                if (text == spokenText) {
                    return SpeechPromptSelection("alert_${directionKey}_$categoryKey", spokenText, usedGeneric = false)
                }
                val importantText = "$spokenText，请注意"
                if (categoryKey in importantCategories && text == importantText) {
                    return SpeechPromptSelection(
                        "alert_${directionKey}_${categoryKey}_important",
                        importantText,
                        usedGeneric = false,
                    )
                }
            }
            for ((kindKey, kindText) in multipleKinds) {
                val spokenText = "$directionText$kindText"
                if (text == spokenText) {
                    return SpeechPromptSelection("alert_${directionKey}_$kindKey", spokenText, usedGeneric = false)
                }
            }
        }
        val genericDirection = directions.firstOrNull { text.startsWith(it.second) } ?: ("center" to "前方")
        return SpeechPromptSelection("alert_${genericDirection.first}_target", "${genericDirection.second}有目标", usedGeneric = true)
    }
}

/** Manages local Chinese speech. All public methods must be called on the main thread. */
class SpeechController(
    context: Context,
    private val onStatus: (String) -> Unit,
    private val onSpoken: (String) -> Unit,
    private val onPlayback: (SpeechPlaybackEvent) -> Unit = {},
    private val directionalAudioConfig: DirectionalAudioConfig = DirectionalAudioConfig(),
) {
    private val appContext = context.applicationContext
    private val mainHandler = Handler(Looper.getMainLooper())
    private val gate = SpeechGate()
    private val audioManager = appContext.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    private val focusListener = AudioManager.OnAudioFocusChangeListener(::onAudioFocusChange)
    private val speechAudioAttributes = AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
        .build()

    private var tts: TextToSpeech? = null
    private var localPlayer: MediaPlayer? = null
    private var localPlaybackId: String? = null
    private var fallbackMode = false
    private var focusRequest: AudioFocusRequest? = null
    private var hasAudioFocus = false
    private var activeUtteranceId: String? = null
    private var activeText: String? = null
    private var activeRequest: SpeechRequest? = null
    private var activePlaybackStarted = false
    private val requestQueue = SpeechRequestQueue()
    private var expiryTimeout: Runnable? = null
    private var stopUtteranceId: String? = null
    private var stopTimeout: Runnable? = null
    private var initializationTimeout: Runnable? = null
    private var sequence = 0L

    val isReady: Boolean
        get() = gate.canSpeak || (fallbackMode && gate.canUseFallback && !gate.isMuted)
    val canQueueOrdinaryAlert: Boolean
        get() = isReady && requestQueue.peek() == null
    val isOfflineTtsReady: Boolean
        get() = gate.canSpeak && tts != null && !fallbackMode
    val canSpeakDynamic: Boolean
        get() = isOfflineTtsReady

    /** Instrumentation-only entry: same lifecycle and scheduler, with local fallback forced. */
    internal fun startLocalFallbackForTest() {
        checkMainThread()
        if (gate.isClosed) return
        releaseEngine("test_session_restarted")
        val session = gate.start()
        gate.initializationFinished(session, succeeded = false)
        fallbackMode = true
        onStatus("固定本地提示音测试已就绪")
    }

    fun start() {
        checkMainThread()
        if (gate.isClosed) return
        releaseEngine("session_restarted")
        fallbackMode = false
        val session = gate.start()
        onStatus("正在初始化离线中文语音…")
        scheduleInitializationTimeout(session)
        try {
            val engine = TextToSpeech(appContext) { result ->
                mainHandler.post { handleInitialization(session, result) }
            }
            tts = engine
            engine.setOnUtteranceProgressListener(progressListener(session))
        } catch (error: RuntimeException) {
            if (gate.isCurrentInitialization(session)) {
                cancelInitializationTimeout()
                gate.initializationFinished(session, succeeded = false)
                tts?.shutdown()
                tts = null
                Log.e("SG_TTS", "TextToSpeech construction failed; using local prompts", error)
                enableLocalFallback(session)
            }
        }
    }

    fun speak(text: String, important: Boolean = false) {
        offer(text, important, eventId = null, validUntilElapsedMs = Long.MAX_VALUE)
    }

    /** Cancels current and waiting alerts while keeping the initialized speech engine ready. */
    fun cancelAlerts(reason: String = "input_paused") {
        checkMainThread()
        cancelQueued(reason)
        cancelCurrent(reason, keepAudioFocus = false)
        abandonAudioFocus()
    }

    /** Cancel only obsolete semantic requests; preserve YOLO/system speech. */
    fun cancelSemanticExcept(validIds: Set<String>, reason: String) {
        checkMainThread()
        fun obsolete(request: SpeechRequest?) = request?.eventId?.let { it.startsWith("semantic:") && it !in validIds } == true
        if (obsolete(requestQueue.peek())) cancelQueued(reason)
        // Once a fresh utterance starts, normal TTL does not truncate it; scene/session changes do.
        if (obsolete(activeRequest) && (!activePlaybackStarted || reason != "semantic_expired")) {
            cancelCurrent(reason, keepAudioFocus = true); startNextQueuedRequest()
        }
    }

    /** Returns true when this request is now active or occupies the single waiting slot. */
    fun cancelResearch(reason: String) {
        checkMainThread()
        if(requestQueue.peek()?.eventId?.startsWith("research:")==true) cancelQueued(reason)
        if(activeRequest?.eventId?.startsWith("research:")==true && (!activePlaybackStarted || reason!="research_expired")) {
            cancelCurrent(reason,keepAudioFocus=true);startNextQueuedRequest()
        }
    }

    fun offer(
        text: String,
        important: Boolean,
        eventId: String?,
        validUntilElapsedMs: Long,
        direction: Direction? = null,
        requireExactSpeech: Boolean = false,
    ): Boolean {
        checkMainThread()
        val request = SpeechRequest(text, important, eventId, validUntilElapsedMs, direction, requireExactSpeech)
        record(SpeechPlaybackPhase.REQUESTED, request)
        if (text.isBlank()) return suppress(request, "blank_text")
        if (SpeechRequestRules.isExpired(request, SystemClock.elapsedRealtime())) return suppress(request, "expired")
        if (!canAcceptRequest()) return suppress(request, rejectionReason())
        if (requireExactSpeech && !canSpeakDynamic) return suppress(request, "dynamic_speech_unavailable")
        discardExpiredQueuedRequest()

        val current = activeRequest
        if (current == null) return startRequest(request)

        if (SpeechRequestRules.preempts(request, current)) {
            cancelCurrent("preempted_by_important", keepAudioFocus = true)
            return startRequest(request)
        }

        val queueOffer = requestQueue.offer(request)
        if (!queueOffer.accepted) return suppress(request, "waiting_slot_full")
        queueOffer.replaced?.let { replaced ->
            record(SpeechPlaybackPhase.CANCELLED, replaced, detail = "replaced_by_important")
        }
        record(SpeechPlaybackPhase.ACCEPTED, request, detail = "queued")
        return true
    }

    fun setMuted(value: Boolean) {
        checkMainThread()
        gate.setMuted(value)
        onStatus(
            when {
                value -> "语音提醒已静音"
                fallbackMode && gate.canUseFallback -> "缺少离线中文TTS，使用固定本地提示音"
                gate.isReady -> "语音提醒已开启"
                gate.isInitializing -> "正在初始化离线中文语音…"
                else -> "无可用的离线中文 TTS，语音提醒不会联网"
            },
        )
        if (value) {
            cancelQueued("muted")
            cancelCurrent("muted", keepAudioFocus = false)
            if (gate.isStopping) finishStop() else abandonAudioFocus()
        }
    }

    fun stop(announce: Boolean = true) {
        checkMainThread()
        if (gate.isClosed || gate.isStopping) return
        val canAnnounce = announce && gate.canSpeak && !gate.isMuted
        val canAnnounceFallback = announce && fallbackMode && gate.canUseFallback && !gate.isMuted
        cancelQueued("session_end")
        cancelCurrent("session_end", keepAudioFocus = true)
        gate.beginStop()
        val engine = tts
        onStatus("感知已停止")
        if (canAnnounceFallback && requestAudioFocus()) {
            startStopAnnouncement("感知已停止", fallback = true)
            return
        }
        if (canAnnounce && engine != null && requestAudioFocus()) {
            startStopAnnouncement("感知已停止", fallback = false)
        } else {
            finishStop()
        }
    }

    fun close() {
        checkMainThread()
        if (gate.isClosed) return
        cancelQueued("closed")
        cancelCurrent("closed", keepAudioFocus = false)
        gate.close()
        releaseEngine("closed")
        onStatus("语音提醒已关闭")
    }

    private fun handleInitialization(session: Long, result: Int) {
        checkMainThread()
        val engine = tts
        if (engine == null || !gate.isCurrentInitialization(session)) return
        cancelInitializationTimeout()
        if (!gate.initializationFinished(session, result == TextToSpeech.SUCCESS)) {
            engine.shutdown()
            tts = null
            enableLocalFallback(session)
            return
        }

        val voice = findOfflineChineseVoice(engine)
        if (voice == null || engine.setVoice(voice) != TextToSpeech.SUCCESS) {
            gate.fail(session)
            engine.shutdown()
            tts = null
            enableLocalFallback(session)
            return
        }

        if (engine.setAudioAttributes(speechAudioAttributes) != TextToSpeech.SUCCESS) {
            gate.fail(session)
            engine.shutdown()
            tts = null
            enableLocalFallback(session)
            return
        }
        Log.i(
            "SG_TTS",
            "selectedVoice name=${voice.name} locale=${voice.locale.toLanguageTag()} " +
                "networkRequired=${voice.isNetworkConnectionRequired}",
        )
        engine.setOnUtteranceProgressListener(progressListener(session))
        onStatus("离线中文语音已就绪")
        speak("感知已开启", important = true)
    }

    private fun canAcceptRequest(): Boolean =
        !gate.isClosed && !gate.isStopping && !gate.isMuted &&
            (gate.canSpeak || (fallbackMode && gate.canUseFallback))

    private fun rejectionReason(): String = when {
        gate.isClosed -> "closed"
        gate.isStopping -> "session_stopping"
        gate.isMuted -> "muted"
        gate.isInitializing -> "initializing"
        else -> "not_ready"
    }

    private fun startRequest(request: SpeechRequest): Boolean {
        if (SystemClock.elapsedRealtime() >= request.validUntilElapsedMs) {
            return suppress(request, "expired")
        }
        if (!canAcceptRequest()) return suppress(request, rejectionReason())
        if (request.requireExactSpeech && !canSpeakDynamic) return suppress(request, "dynamic_speech_unavailable")
        if (!requestAudioFocus()) {
            onStatus(if (fallbackMode) "未获得音频焦点，固定提示音已取消" else "未获得音频焦点，提醒已取消")
            return suppress(request, "audio_focus_denied")
        }

        activeRequest = request
        activePlaybackStarted = false
        record(SpeechPlaybackPhase.ACCEPTED, request, detail = "started")
        scheduleExpiryTimeout(request)
        if (fallbackMode && gate.canUseFallback) {
            playLocalPrompt(request, isStopAnnouncement = false)
            return true
        }
        val engine = tts
        if (engine == null) {
            clearActiveRequest()
            abandonAudioFocus()
            return suppress(request, "tts_unavailable")
        }
        enqueue(engine, request, isStopAnnouncement = false)
        return true
    }

    private fun startStopAnnouncement(text: String, fallback: Boolean) {
        val request = SpeechRequest(text, important = true, eventId = null, validUntilElapsedMs = Long.MAX_VALUE)
        record(SpeechPlaybackPhase.REQUESTED, request, detail = "session_end_announcement")
        record(SpeechPlaybackPhase.ACCEPTED, request, detail = "session_end_announcement")
        activeRequest = request
        activePlaybackStarted = false
        scheduleExpiryTimeout(request)
        if (fallback) {
            playLocalPrompt(request, isStopAnnouncement = true)
        } else {
            val engine = tts
            if (engine == null) finishStop() else enqueue(engine, request, isStopAnnouncement = true)
        }
    }

    private fun suppress(request: SpeechRequest, reason: String): Boolean {
        record(SpeechPlaybackPhase.SUPPRESSED, request, detail = reason)
        return false
    }

    private fun record(
        phase: SpeechPlaybackPhase,
        request: SpeechRequest,
        spokenText: String? = null,
        detail: String? = null,
        resourceName: String? = null,
        requestedText: String = request.text,
    ) {
        val event = SpeechPlaybackEvent(
            phase = phase,
            eventId = request.eventId,
            requestedText = requestedText,
            spokenText = spokenText,
            important = request.important,
            elapsedRealtimeMs = SystemClock.elapsedRealtime(),
            validUntilElapsedMs = request.validUntilElapsedMs,
            detail = detail,
            resourceName = resourceName,
            sessionGeneration = gate.generation,
            direction = request.direction,
        )
        Log.i(
            "SG_SPEECH",
            "event=${phase.logName} eventId=${event.eventId ?: "none"} important=${event.important} " +
                "requestText=${event.requestedText} spokenText=${event.spokenText ?: "none"} " +
                "elapsedMs=${event.elapsedRealtimeMs} validUntilMs=${event.validUntilElapsedMs} " +
                "detail=${event.detail ?: "none"} resource=${event.resourceName ?: "none"} " +
                "direction=${event.direction ?: "none"}",
        )
        mainHandler.post {
            try {
                onPlayback(event)
            } catch (error: RuntimeException) {
                Log.e("SG_SPEECH", "playback observer failed", error)
            }
        }
    }

    private fun scheduleExpiryTimeout(request: SpeechRequest) {
        cancelExpiryTimeout()
        if (request.validUntilElapsedMs == Long.MAX_VALUE) return
        val timeout = Runnable {
            if (activeRequest !== request || activePlaybackStarted) return@Runnable
            if (SystemClock.elapsedRealtime() >= request.validUntilElapsedMs) {
                cancelCurrent("expired_before_start", keepAudioFocus = true)
                startNextQueuedRequest()
            } else {
                scheduleExpiryTimeout(request)
            }
        }
        expiryTimeout = timeout
        mainHandler.postDelayed(timeout, (request.validUntilElapsedMs - SystemClock.elapsedRealtime()).coerceAtLeast(0L))
    }

    private fun cancelExpiryTimeout() {
        expiryTimeout?.let(mainHandler::removeCallbacks)
        expiryTimeout = null
    }

    private fun startNextQueuedRequest() {
        val request = requestQueue.take() ?: run {
            abandonAudioFocus()
            return
        }
        if (SystemClock.elapsedRealtime() >= request.validUntilElapsedMs) {
            suppress(request, "expired_while_queued")
            startNextQueuedRequest()
            return
        }
        if (!canAcceptRequest()) {
            record(SpeechPlaybackPhase.CANCELLED, request, detail = rejectionReason())
            abandonAudioFocus()
            return
        }
        startRequest(request)
    }

    private fun discardExpiredQueuedRequest() {
        requestQueue.discardExpired(SystemClock.elapsedRealtime())?.let { waiting ->
            suppress(waiting, "expired_while_queued")
        }
    }

    private fun cancelQueued(reason: String) {
        requestQueue.clear()?.let { request ->
            record(SpeechPlaybackPhase.CANCELLED, request, detail = reason)
        }
    }

    private fun cancelCurrent(reason: String, keepAudioFocus: Boolean) {
        val request = activeRequest
        if (request != null) record(SpeechPlaybackPhase.CANCELLED, request, detail = reason)
        cancelExpiryTimeout()
        stopLocalPlayback()
        tts?.stop()
        clearActiveUtterance()
        clearActiveRequest()
        if (!keepAudioFocus) abandonAudioFocus()
    }

    private fun clearActiveRequest() {
        activeRequest = null
        activePlaybackStarted = false
    }

    private fun enableLocalFallback(session: Long) {
        if (session != gate.generation || gate.isClosed) return
        fallbackMode = true
        onStatus("缺少离线中文TTS，使用固定本地提示音")
        startRequest(
            SpeechRequest("感知已开启", important = true, eventId = null, validUntilElapsedMs = Long.MAX_VALUE),
        )
    }

    private fun playLocalPrompt(request: SpeechRequest, isStopAnnouncement: Boolean) {
        if (request.requireExactSpeech) {
            onStatus("无可用的离线动态语音，请查看环境概览文本")
            cancelCurrent("dynamic_speech_unavailable", keepAudioFocus = !isStopAnnouncement)
            if (isStopAnnouncement) finishStop() else startNextQueuedRequest()
            return
        }
        val reason = when {
            gate.isMuted -> "muted"
            gate.isClosed -> "closed"
            gate.isStopping && !isStopAnnouncement -> "session_stopping"
            SystemClock.elapsedRealtime() >= request.validUntilElapsedMs -> "expired"
            else -> null
        }
        if (reason != null) {
            cancelCurrent(reason, keepAudioFocus = !isStopAnnouncement)
            if (isStopAnnouncement) finishStop() else startNextQueuedRequest()
            return
        }
        val prompt = resolveLocalPrompt(request.text)
        val session = gate.generation
        val id = "local-$session-${++sequence}-${UUID.randomUUID()}"
        localPlaybackId = id
        if (isStopAnnouncement) {
            stopUtteranceId = id
            scheduleStopTimeout(id, session)
        }
        val player = MediaPlayer()
        localPlayer = player
        try {
            val descriptor = appContext.resources.openRawResourceFd(prompt.resourceId)
                ?: error("Missing local prompt resource ${prompt.resourceId}")
            descriptor.use {
                player.setDataSource(it.fileDescriptor, it.startOffset, it.length)
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                player.setAudioAttributes(speechAudioAttributes)
            }
            player.setOnPreparedListener { prepared ->
                if (localPlayer !== prepared || localPlaybackId != id || session != gate.generation ||
                    gate.isMuted || (gate.isStopping && !isStopAnnouncement)
                ) return@setOnPreparedListener
                if (SystemClock.elapsedRealtime() >= request.validUntilElapsedMs) {
                    cancelCurrent("expired_before_start", keepAudioFocus = !isStopAnnouncement)
                    if (isStopAnnouncement) finishStop() else startNextQueuedRequest()
                    return@setOnPreparedListener
                }
                try {
                    val mix = DirectionalAudio.mix(request.direction, directionalAudioConfig)
                    prepared.setVolume(mix.leftGain, mix.rightGain)
                    prepared.start()
                    if (prepared.isPlaying) {
                        activePlaybackStarted = true
                        cancelExpiryTimeout()
                        record(
                            SpeechPlaybackPhase.ON_START,
                            request,
                            spokenText = prompt.spokenText,
                            resourceName = prompt.resourceName,
                            detail = "local_media_player",
                        )
                        onSpoken(prompt.spokenText)
                    }
                } catch (error: RuntimeException) {
                    Log.e("SG_LOCAL_AUDIO", "start failed resource=${prompt.resourceName}", error)
                    onStatus("固定本地提示音播放失败")
                    cancelCurrent("local_start_failed", keepAudioFocus = !isStopAnnouncement)
                    if (isStopAnnouncement) finishStop() else startNextQueuedRequest()
                }
            }
            player.setOnCompletionListener { completed ->
                if (session != gate.generation || localPlayer !== completed || localPlaybackId != id) return@setOnCompletionListener
                record(SpeechPlaybackPhase.COMPLETED, request, spokenText = prompt.spokenText,
                    resourceName = prompt.resourceName, detail = "local_media_player")
                stopLocalPlayback(completed)
                clearActiveRequest()
                cancelExpiryTimeout()
                if (isStopAnnouncement) finishStop() else startNextQueuedRequest()
            }
            player.setOnErrorListener { failed, what, extra ->
                if (session == gate.generation && localPlayer === failed && localPlaybackId == id) {
                    Log.e("SG_LOCAL_AUDIO", "playback error what=$what extra=$extra resource=${prompt.resourceName}")
                    onStatus("固定本地提示音播放失败")
                    cancelCurrent("local_playback_error", keepAudioFocus = !isStopAnnouncement)
                    if (isStopAnnouncement) finishStop() else startNextQueuedRequest()
                }
                true
            }
            player.prepareAsync()
        } catch (error: Exception) {
            Log.e("SG_LOCAL_AUDIO", "prepare failed resource=${prompt.resourceName}", error)
            onStatus("固定本地提示音播放失败")
            cancelCurrent("local_prepare_failed", keepAudioFocus = !isStopAnnouncement)
            if (isStopAnnouncement) finishStop() else startNextQueuedRequest()
        }
    }

    private fun resolveLocalPrompt(text: String): LocalPrompt {
        val selection = SpeechPromptCatalog.resolve(text)
        val resourceId = promptResourceByName(selection.resourceName)
        if (selection.usedGeneric) {
            Log.i("SG_LOCAL_AUDIO", "no exact local clip for '$text'; using generic '${selection.spokenText}'")
        }
        return LocalPrompt(resourceId, selection.spokenText, selection.resourceName)
    }

    private fun localPromptResource(direction: String, category: String): Int = when (direction) {
        "left" -> when (category) {
            "pedestrian" -> R.raw.alert_left_pedestrian
            "bicycle" -> R.raw.alert_left_bicycle
            "car" -> R.raw.alert_left_car
            "motorcycle" -> R.raw.alert_left_motorcycle
            "bus" -> R.raw.alert_left_bus
            "truck" -> R.raw.alert_left_truck
            "dog" -> R.raw.alert_left_dog
            "chair" -> R.raw.alert_left_chair
            "traffic_light" -> R.raw.alert_left_traffic_light
            "stop_sign" -> R.raw.alert_left_stop_sign
            "train" -> R.raw.alert_left_train
            else -> R.raw.alert_left_target
        }
        "right" -> when (category) {
            "pedestrian" -> R.raw.alert_right_pedestrian
            "bicycle" -> R.raw.alert_right_bicycle
            "car" -> R.raw.alert_right_car
            "motorcycle" -> R.raw.alert_right_motorcycle
            "bus" -> R.raw.alert_right_bus
            "truck" -> R.raw.alert_right_truck
            "dog" -> R.raw.alert_right_dog
            "chair" -> R.raw.alert_right_chair
            "traffic_light" -> R.raw.alert_right_traffic_light
            "stop_sign" -> R.raw.alert_right_stop_sign
            "train" -> R.raw.alert_right_train
            else -> R.raw.alert_right_target
        }
        else -> when (category) {
            "pedestrian" -> R.raw.alert_center_pedestrian
            "bicycle" -> R.raw.alert_center_bicycle
            "car" -> R.raw.alert_center_car
            "motorcycle" -> R.raw.alert_center_motorcycle
            "bus" -> R.raw.alert_center_bus
            "truck" -> R.raw.alert_center_truck
            "dog" -> R.raw.alert_center_dog
            "chair" -> R.raw.alert_center_chair
            "traffic_light" -> R.raw.alert_center_traffic_light
            "stop_sign" -> R.raw.alert_center_stop_sign
            "train" -> R.raw.alert_center_train
            else -> R.raw.alert_center_target
        }
    }

    private fun importantPromptResource(direction: String, category: String): Int? = when (direction) {
        "left" -> when (category) {
            "pedestrian" -> R.raw.alert_left_pedestrian_important
            "bicycle" -> R.raw.alert_left_bicycle_important
            "car" -> R.raw.alert_left_car_important
            "motorcycle" -> R.raw.alert_left_motorcycle_important
            "bus" -> R.raw.alert_left_bus_important
            "truck" -> R.raw.alert_left_truck_important
            "traffic_light" -> R.raw.alert_left_traffic_light_important
            "stop_sign" -> R.raw.alert_left_stop_sign_important
            "target" -> R.raw.alert_left_target_important
            else -> null
        }
        "right" -> when (category) {
            "pedestrian" -> R.raw.alert_right_pedestrian_important
            "bicycle" -> R.raw.alert_right_bicycle_important
            "car" -> R.raw.alert_right_car_important
            "motorcycle" -> R.raw.alert_right_motorcycle_important
            "bus" -> R.raw.alert_right_bus_important
            "truck" -> R.raw.alert_right_truck_important
            "traffic_light" -> R.raw.alert_right_traffic_light_important
            "stop_sign" -> R.raw.alert_right_stop_sign_important
            "target" -> R.raw.alert_right_target_important
            else -> null
        }
        else -> when (category) {
            "pedestrian" -> R.raw.alert_center_pedestrian_important
            "bicycle" -> R.raw.alert_center_bicycle_important
            "car" -> R.raw.alert_center_car_important
            "motorcycle" -> R.raw.alert_center_motorcycle_important
            "bus" -> R.raw.alert_center_bus_important
            "truck" -> R.raw.alert_center_truck_important
            "traffic_light" -> R.raw.alert_center_traffic_light_important
            "stop_sign" -> R.raw.alert_center_stop_sign_important
            "target" -> R.raw.alert_center_target_important
            else -> null
        }
    }

    private fun promptResourceByName(name: String): Int {
        when (name) {
            "ar_wall_near" -> return R.raw.ar_wall_near
            "ar_wall_approaching" -> return R.raw.ar_wall_approaching
            "ar_corridor_left" -> return R.raw.ar_corridor_left
            "ar_corridor_center" -> return R.raw.ar_corridor_center
            "ar_corridor_right" -> return R.raw.ar_corridor_right
            "perception_started" -> return R.raw.perception_started
            "perception_stopped" -> return R.raw.perception_stopped
            "perception_interrupted" -> return R.raw.perception_interrupted
            "camera_connection_failed" -> return R.raw.camera_connection_failed
            "perception_limited" -> return R.raw.perception_limited
            "perception_recovered" -> return R.raw.perception_recovered
            "camera_stopped" -> return R.raw.camera_stopped
            "environment_wall" -> return R.raw.environment_wall
            "environment_stairs" -> return R.raw.environment_stairs
            "environment_step" -> return R.raw.environment_step
            "environment_pole" -> return R.raw.environment_pole
            "alert_left_risk_elevated" -> return R.raw.alert_left_risk_elevated
            "alert_center_risk_elevated" -> return R.raw.alert_center_risk_elevated
            "alert_right_risk_elevated" -> return R.raw.alert_right_risk_elevated
            "alert_left_multiple_pedestrians" -> return R.raw.alert_left_multiple_pedestrians
            "alert_center_multiple_pedestrians" -> return R.raw.alert_center_multiple_pedestrians
            "alert_right_multiple_pedestrians" -> return R.raw.alert_right_multiple_pedestrians
            "alert_left_multiple_targets" -> return R.raw.alert_left_multiple_targets
            "alert_center_multiple_targets" -> return R.raw.alert_center_multiple_targets
            "alert_right_multiple_targets" -> return R.raw.alert_right_multiple_targets
        }
        if(name.startsWith("environment_")) {
            val id=appContext.resources.getIdentifier(name,"raw",appContext.packageName)
            require(id!=0) { "Missing exact environment audio: $name" }
            return id
        }
        val body = name.removePrefix("alert_")
        val direction = body.substringBefore('_')
        val categoryAndImportance = body.substringAfter('_')
        return if (categoryAndImportance.endsWith("_important")) {
            val category = categoryAndImportance.removeSuffix("_important")
            importantPromptResource(direction, category) ?: R.raw.alert_center_target
        } else {
            localPromptResource(direction, categoryAndImportance)
        }
    }

    private fun findOfflineChineseVoice(engine: TextToSpeech): Voice? {
        return try {
            engine.voices
                ?.asSequence()
                ?.filter { voice ->
                    voice.locale.language == "zh" &&
                        !voice.isNetworkConnectionRequired &&
                        voice.features?.contains(TextToSpeech.Engine.KEY_FEATURE_NETWORK_SYNTHESIS) != true &&
                        voice.features?.contains(TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED) != true
                }
                ?.sortedWith(
                    compareByDescending<Voice> { it.locale.country == "CN" }
                        .thenByDescending { it.quality },
                )
                ?.firstOrNull()
        } catch (_: RuntimeException) {
            null
        }
    }

    private fun enqueue(engine: TextToSpeech, request: SpeechRequest, isStopAnnouncement: Boolean) {
        if (SystemClock.elapsedRealtime() >= request.validUntilElapsedMs) {
            cancelCurrent("expired_before_enqueue", keepAudioFocus = !isStopAnnouncement)
            if (isStopAnnouncement) finishStop() else startNextQueuedRequest()
            return
        }
        val id = "sg-${gate.generation}-${++sequence}-${UUID.randomUUID()}"
        activeUtteranceId = id
        activeText = request.text
        if (isStopAnnouncement) stopUtteranceId = id
        val parameters = Bundle().apply {
            putFloat(TextToSpeech.Engine.KEY_PARAM_PAN, DirectionalAudio.mix(request.direction, directionalAudioConfig).ttsPan)
        }
        val result = engine.speak(request.text, TextToSpeech.QUEUE_FLUSH, parameters, id)
        if (result != TextToSpeech.SUCCESS) {
            fallbackMode = true
            onStatus(if (request.requireExactSpeech) "离线动态语音播放失败，请查看环境概览文本" else "离线中文TTS播放失败，使用固定本地提示音")
            clearActiveUtterance()
            engine.shutdown()
            tts = null
            playLocalPrompt(request, isStopAnnouncement)
        } else if (isStopAnnouncement) {
            scheduleStopTimeout(id, gate.generation)
        }
    }

    private fun scheduleStopTimeout(utteranceId: String, session: Long) {
        stopTimeout?.let(mainHandler::removeCallbacks)
        val timeout = Runnable {
            if (gate.generation == session && gate.isStopping && stopUtteranceId == utteranceId) {
                onStatus("停止提示超时，语音引擎已关闭")
                finishStop()
            }
        }
        stopTimeout = timeout
        mainHandler.postDelayed(timeout, STOP_ANNOUNCEMENT_TIMEOUT_MS)
    }

    private fun scheduleInitializationTimeout(session: Long) {
        cancelInitializationTimeout()
        val timeout = Runnable {
            if (!gate.isCurrentInitialization(session)) return@Runnable
            gate.initializationFinished(session, succeeded = false)
            Log.w("SG_TTS", "initialization timed out after ${TTS_INITIALIZATION_TIMEOUT_MS}ms; using local prompts")
            tts?.shutdown()
            tts = null
            enableLocalFallback(session)
        }
        initializationTimeout = timeout
        mainHandler.postDelayed(timeout, TTS_INITIALIZATION_TIMEOUT_MS)
    }

    private fun cancelInitializationTimeout() {
        initializationTimeout?.let(mainHandler::removeCallbacks)
        initializationTimeout = null
    }

    private fun progressListener(session: Long) = object : UtteranceProgressListener() {
        override fun onStart(utteranceId: String?) {
            mainHandler.post {
                val id = utteranceId ?: return@post
                val allowed = id == activeUtteranceId &&
                    (gate.canSpeak || (id == stopUtteranceId && gate.isStopping && !gate.isMuted))
                if (session != gate.generation || !allowed) return@post
                val request = activeRequest ?: return@post
                if (!activePlaybackStarted && SystemClock.elapsedRealtime() >= request.validUntilElapsedMs) {
                    cancelCurrent("expired_before_start", keepAudioFocus = id != stopUtteranceId)
                    if (id == stopUtteranceId) finishStop() else startNextQueuedRequest()
                    return@post
                }
                val text = activeText ?: return@post
                activePlaybackStarted = true
                cancelExpiryTimeout()
                record(SpeechPlaybackPhase.ON_START, request, spokenText = text, detail = "android_tts")
                onSpoken(text)
            }
        }

        override fun onDone(utteranceId: String?) {
            mainHandler.post {
                val id = utteranceId ?: return@post
                if (id != activeUtteranceId || session != gate.generation) return@post
                activeRequest?.let { record(SpeechPlaybackPhase.COMPLETED, it, spokenText = activeText, detail = "android_tts") }
                if (id == stopUtteranceId) {
                    clearActiveUtterance()
                    clearActiveRequest()
                    cancelExpiryTimeout()
                    finishStop()
                } else {
                    clearActiveUtterance()
                    clearActiveRequest()
                    cancelExpiryTimeout()
                    startNextQueuedRequest()
                }
            }
        }

        @Deprecated("Deprecated by Android; retained for engines using the legacy callback")
        override fun onError(utteranceId: String?) = handleUtteranceError(session, utteranceId)

        override fun onError(utteranceId: String?, errorCode: Int) =
            handleUtteranceError(session, utteranceId)

        override fun onStop(utteranceId: String?, interrupted: Boolean) {
            mainHandler.post {
                val id = utteranceId ?: return@post
                if (session != gate.generation) return@post
                if (id == stopUtteranceId && gate.isStopping) {
                    clearActiveUtterance()
                    clearActiveRequest()
                    cancelExpiryTimeout()
                    finishStop()
                } else if (id == activeUtteranceId) {
                    activeRequest?.let { record(SpeechPlaybackPhase.CANCELLED, it, detail = "tts_stopped") }
                    clearActiveUtterance()
                    clearActiveRequest()
                    cancelExpiryTimeout()
                    if (gate.isMuted) abandonAudioFocus() else startNextQueuedRequest()
                }
            }
        }
    }

    private fun handleUtteranceError(session: Long, utteranceId: String?) {
        mainHandler.post {
            val id = utteranceId ?: return@post
            if (session != gate.generation || id != activeUtteranceId) return@post
            val failedRequest = activeRequest ?: return@post
            val isStopAnnouncement = id == stopUtteranceId
            fallbackMode = true
            onStatus(if (failedRequest.requireExactSpeech) "离线动态语音播放失败，请查看环境概览文本" else "离线中文TTS播放失败，使用固定本地提示音")
            clearActiveUtterance()
            tts?.shutdown()
            tts = null
            playLocalPrompt(failedRequest, isStopAnnouncement)
        }
    }

    private fun requestAudioFocus(): Boolean {
        if (hasAudioFocus) return true
        val touchExploration=(appContext.getSystemService(Context.ACCESSIBILITY_SERVICE) as android.view.accessibility.AccessibilityManager).isTouchExplorationEnabled
        Log.i("SG_AUDIO","focus requested policy=${if(touchExploration) "transient_may_duck" else "transient"}")
        val result = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val request = AudioFocusRequest.Builder(if (touchExploration) AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK else AudioManager.AUDIOFOCUS_GAIN_TRANSIENT)
                .setAudioAttributes(speechAudioAttributes)
                .setOnAudioFocusChangeListener(focusListener, mainHandler)
                .build()
            focusRequest = request
            audioManager.requestAudioFocus(request)
        } else {
            @Suppress("DEPRECATION")
            audioManager.requestAudioFocus(
                focusListener,
                AudioManager.STREAM_MUSIC,
                AudioManager.AUDIOFOCUS_GAIN_TRANSIENT,
            )
        }
        hasAudioFocus = result == AudioManager.AUDIOFOCUS_REQUEST_GRANTED
        return hasAudioFocus
    }

    private fun onAudioFocusChange(change: Int) {
        checkMainThread()
        if (change == AudioManager.AUDIOFOCUS_GAIN) return
        cancelQueued("audio_focus_lost")
        cancelCurrent("audio_focus_lost", keepAudioFocus = false)
        hasAudioFocus = false
        focusRequest = null
        onStatus("音频焦点丢失，提醒已取消")
        if (gate.isStopping) finishStop() else abandonAudioFocus()
    }

    private fun abandonAudioFocus() {
        if (!hasAudioFocus) return
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            focusRequest?.let(audioManager::abandonAudioFocusRequest)
            focusRequest = null
        } else {
            @Suppress("DEPRECATION")
            audioManager.abandonAudioFocus(focusListener)
        }
        hasAudioFocus = false
    }

    private fun finishStop() {
        stopTimeout?.let(mainHandler::removeCallbacks)
        stopTimeout = null
        cancelInitializationTimeout()
        cancelQueued("session_stopped")
        if (activeRequest != null) cancelCurrent("session_stopped", keepAudioFocus = false)
        stopLocalPlayback()
        clearActiveUtterance()
        abandonAudioFocus()
        tts?.shutdown()
        tts = null
        stopUtteranceId = null
        fallbackMode = false
        gate.finishStop()
    }

    private fun releaseEngine(reason: String) {
        stopTimeout?.let(mainHandler::removeCallbacks)
        stopTimeout = null
        cancelInitializationTimeout()
        cancelQueued(reason)
        cancelCurrent(reason, keepAudioFocus = false)
        stopLocalPlayback()
        tts?.stop()
        tts?.shutdown()
        tts = null
        clearActiveUtterance()
        stopUtteranceId = null
        abandonAudioFocus()
    }

    private fun stopLocalPlayback(expectedPlayer: MediaPlayer? = null) {
        val player = localPlayer ?: return
        if (expectedPlayer != null && expectedPlayer !== player) return
        localPlayer = null
        localPlaybackId = null
        try {
            if (player.isPlaying) player.stop()
        } catch (_: IllegalStateException) {
            // A player that is still preparing cannot be stopped; release() still cancels it.
        }
        player.release()
    }

    private fun clearActiveUtterance() {
        activeUtteranceId = null
        activeText = null
    }

    private fun checkMainThread() {
        check(Looper.myLooper() == Looper.getMainLooper()) {
            "SpeechController methods must be called on the main thread"
        }
    }

    private companion object {
        const val STOP_ANNOUNCEMENT_TIMEOUT_MS = 3_000L
        const val TTS_INITIALIZATION_TIMEOUT_MS = 8_000L
    }

    private data class LocalPrompt(
        val resourceId: Int,
        val spokenText: String,
        val resourceName: String,
    )
}

/** Pure lifecycle gate shared with local unit tests. */
internal class SpeechGate {
    private enum class State { IDLE, INITIALIZING, READY, FAILED, STOPPING, CLOSED }

    private var state = State.IDLE
    private var muted = false
    var generation: Long = 0
        private set

    val isMuted: Boolean get() = muted
    val isClosed: Boolean get() = state == State.CLOSED
    val isStopping: Boolean get() = state == State.STOPPING
    val isReady: Boolean get() = state == State.READY
    val isInitializing: Boolean get() = state == State.INITIALIZING
    val isFallbackReady: Boolean get() = state == State.READY || state == State.FAILED
    val canUseFallback: Boolean get() = isFallbackReady && !muted
    val canSpeak: Boolean get() = state == State.READY && !muted

    fun start(): Long {
        if (state == State.CLOSED) return generation
        generation += 1
        state = State.INITIALIZING
        return generation
    }

    fun isCurrentInitialization(token: Long): Boolean =
        token == generation && state == State.INITIALIZING

    fun initializationFinished(token: Long, succeeded: Boolean): Boolean {
        if (!isCurrentInitialization(token)) return false
        state = if (succeeded) State.READY else State.FAILED
        return succeeded
    }

    fun fail(token: Long) {
        if (token == generation && state == State.READY) state = State.FAILED
    }

    fun setMuted(value: Boolean) {
        muted = value
    }

    fun beginStop() {
        if (state != State.CLOSED) state = State.STOPPING
    }

    fun finishStop() {
        if (state == State.STOPPING) state = State.IDLE
    }

    fun close() {
        state = State.CLOSED
        generation += 1
    }
}
