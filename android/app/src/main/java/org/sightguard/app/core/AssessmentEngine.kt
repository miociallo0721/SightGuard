package org.sightguard.app.core

import kotlin.math.hypot

enum class EventState { CANDIDATE, ACTIVE, LOST, RESOLVED }

data class EventSnapshot(
    val eventId: String, val state: EventState, val classId: Int, val className: String,
    val firstSeenMs: Double, val lastSeenMs: Double, val validObservations: Int,
    val confirmed: Boolean, val sourceTrackIds: List<Long>, val lastTrackId: Long?,
    val observedNow: Boolean, val assessment: RiskAssessment?,
)
data class EventChange(
    val streamId: String, val eventId: String, val timestampMs: Double,
    val eventKind: String, val action: String, val state: EventState, val reason: String,
    val sourceTrackId: Long? = null, val associationMethod: String? = null,
    val snapshot: EventSnapshot? = null,
)
data class AlertEvent(
    val streamId: String, val alertId: String, val eventId: String, val timestampMs: Double,
    val riskLevel: RiskLevel?, val riskScore: Double?, val alertType: String, val reason: String,
    val sourceTrackIds: List<Long>, val qualityScore: Double? = null,
)
data class PerceptionStatus(
    val status: String = "available", val eventId: String? = null,
    val state: EventState? = null, val sinceMs: Double? = null, val reason: String? = null,
)
data class RiskObservation(
    val detectionIndex: Int, val sourceTrackId: Long?, val classId: Int, val className: String,
    val eventId: String?, val associationMethod: String?, val features: RiskFeatures?,
    val assessment: RiskAssessment?, val error: String? = null,
)
data class RiskFrame(
    val streamId: String, val frameIndex: Int, val timestampMs: Double,
    val timestampSource: String, val inputStatus: String, val perception: PerceptionStatus,
    val observations: List<RiskObservation>, val events: List<EventSnapshot>, val warnings: List<String>,
)
data class StepResult(val frame: RiskFrame, val changes: List<EventChange>, val alerts: List<AlertEvent>)

/** PC event semantics; deterministic mutual-best association abstains on ambiguity. */
class EventManager(private val config: AssociationConfig, private val streamId: String) {
    internal data class MutableEvent(
        val eventId: String, var classId: Int, var className: String, val firstSeenMs: Double,
        var lastSeenMs: Double, var spatial: SpatialFeatures, var lastTrackId: Long?,
        var state: EventState = EventState.CANDIDATE, var confirmed: Boolean = false,
        var validObservations: Int = 0, var confirmationStartMs: Double? = null,
        var confirmationCount: Int = 0, val sourceTrackIds: MutableList<Long> = mutableListOf(),
        var observedNow: Boolean = false, var assessment: RiskAssessment? = null,
    ) {
        fun snapshot(): EventSnapshot = EventSnapshot(eventId, state, classId, className, firstSeenMs,
            lastSeenMs, validObservations, confirmed, sourceTrackIds.toList(), lastTrackId, observedNow, assessment)
    }
    internal data class Association(val detectionIndex: Int, val event: MutableEvent, val method: String, val created: Boolean)
    private data class Edge(val score: Double, val eventId: String, val index: Int)
    private val events = linkedMapOf<String, MutableEvent>()
    private var nextId = 1
    internal fun liveEvents(): List<MutableEvent> = events.values.filter { it.state != EventState.RESOLVED }.sortedBy { it.eventId }
    private fun change(event: MutableEvent, time: Double, action: String, reason: String, method: String? = null): EventChange =
        EventChange(streamId, event.eventId, time, "risk", action, event.state, reason, event.lastTrackId, method, event.snapshot())

    fun prepare(time: Double, complete: Boolean): List<EventChange> {
        val changes = mutableListOf<EventChange>()
        liveEvents().forEach { event ->
            event.observedNow = false
            if (complete && time - event.lastSeenMs > config.recoveryWindowMs) {
                event.state = EventState.RESOLVED
                changes += change(event, time, "resolved", "observation_window_expired")
            }
        }
        // Resolved snapshots are in changes. Keeping them forever is unnecessary on a phone.
        events.entries.removeAll { it.value.state == EventState.RESOLVED }
        return changes
    }

    internal fun associate(frame: PFrame, spatial: Map<Int, SpatialFeatures>): List<Association> {
        val ordered = spatial.keys.sortedWith { a, b ->
            val da = frame.detections[a]; val db = frame.detections[b]
            var result = da.classId.compareTo(db.classId)
            if (result == 0) for (i in 0..3) {
                result = spatial.getValue(a).bboxNormalized[i].compareTo(spatial.getValue(b).bboxNormalized[i])
                if (result != 0) break
            }
            if (result == 0) result = (da.trackId ?: -1).compareTo(db.trackId ?: -1)
            if (result == 0) result = db.confidence.compareTo(da.confidence)
            if (result == 0) result = a.compareTo(b)
            result
        }
        val live = liveEvents()
        val availableIndices = ordered.toMutableSet(); val availableEvents = live.map { it.eventId }.toMutableSet()
        val assigned = mutableMapOf<Int, Pair<MutableEvent, String>>()
        val detectionIds = ordered.mapNotNull { frame.detections[it].trackId }.groupingBy { it }.eachCount()
        val eventIds = live.mapNotNull { it.lastTrackId }.groupingBy { it }.eachCount()
        for (method in listOf("track_id", "geometry")) {
            val edges = mutableListOf<Edge>()
            for (index in ordered) {
                if (index !in availableIndices) continue
                val detection = frame.detections[index]; val geometry = spatial.getValue(index)
                for (event in live) {
                    if (event.eventId !in availableEvents) continue
                    val switched = event.classId != detection.classId
                    if (switched && !(method == "track_id" && config.allowClassSwitchForTrackId)) continue
                    val dt = frame.timestampMs - event.lastSeenMs
                    if (dt < 0 || dt > config.recoveryWindowMs) continue
                    val distance = hypot(geometry.centerXY[0] - event.spatial.centerXY[0], geometry.centerXY[1] - event.spatial.centerXY[1])
                    val overlap = bboxIou(geometry.bboxNormalized, event.spatial.bboxNormalized)
                    val score: Double
                    if (method == "track_id") {
                        if (detection.trackId == null || event.lastTrackId != detection.trackId ||
                            detectionIds[detection.trackId] != 1 || eventIds[detection.trackId] != 1 ||
                            distance > config.trackMaxCenterDistance || (switched && overlap < config.geometryMinIou)) continue
                        score = overlap
                    } else {
                        if (distance > config.geometryMaxCenterDistance || overlap < config.geometryMinIou) continue
                        score = config.iouWeight * overlap + (1 - config.iouWeight) * (1 - distance / config.geometryMaxCenterDistance)
                        if (score < config.geometryMinScore) continue
                    }
                    edges += Edge(score, event.eventId, index)
                }
            }
            val sorted = edges.sortedWith(compareByDescending<Edge> { it.score }.thenBy { it.eventId }.thenBy { it.index })
            val byDetection = sorted.groupBy { it.index }; val byEvent = sorted.groupBy { it.eventId }
            for (edge in sorted) {
                val left = byDetection.getValue(edge.index); val right = byEvent.getValue(edge.eventId)
                if (edge != left.first() || edge != right.first()) continue
                if (listOf(left, right).any { it.size > 1 && edge.score - it[1].score <= config.ambiguityMargin }) continue
                val event = events.getValue(edge.eventId)
                val associationMethod = if (method == "track_id" && event.classId != frame.detections[edge.index].classId) "track_id_class_switch" else method
                assigned[edge.index] = event to associationMethod
                availableIndices.remove(edge.index); availableEvents.remove(edge.eventId)
            }
        }
        return ordered.map { index ->
            val pair = assigned[index]
            if (pair != null) Association(index, pair.first, pair.second, false) else {
                val d = frame.detections[index]
                val event = MutableEvent("E%06d".format(nextId++), d.classId, d.className, frame.timestampMs,
                    frame.timestampMs, spatial.getValue(index), d.trackId)
                events[event.eventId] = event
                Association(index, event, "new", true)
            }
        }
    }

    internal fun observe(association: Association, d: PDetection, spatial: SpatialFeatures, assessment: RiskAssessment, time: Double): List<EventChange> {
        val event = association.event; val wasLost = event.state == EventState.LOST
        if (event.confirmationStartMs == null || time - event.lastSeenMs > config.confirmationMaxGapMs) {
            event.confirmationStartMs = time; event.confirmationCount = 0
        }
        event.confirmationCount++; event.validObservations++; event.lastSeenMs = time; event.spatial = spatial
        if (association.method == "track_id_class_switch") { event.classId = d.classId; event.className = d.className }
        event.lastTrackId = d.trackId; event.observedNow = true; event.assessment = assessment
        d.trackId?.let { if (it !in event.sourceTrackIds) event.sourceTrackIds += it }
        if (wasLost) event.state = if (event.confirmed) EventState.ACTIVE else EventState.CANDIDATE
        val action = if (association.created) "created" else if (wasLost) "recovered" else "updated"
        val changes = mutableListOf(change(event, time, action, "valid_observation", association.method))
        if (!event.confirmed && event.confirmationCount >= config.confirmationObservations &&
            time - event.confirmationStartMs!! >= config.confirmationMs) {
            event.confirmed = true; event.state = EventState.ACTIVE
            changes += change(event, time, "activated", "candidate_confirmed")
        }
        return changes
    }
    fun endFrame(time: Double, complete: Boolean): List<EventChange> = liveEvents().mapNotNull { event ->
        if (event.observedNow) null else {
            event.confirmationStartMs = null; event.confirmationCount = 0
            if (complete && event.state != EventState.LOST) {
                event.state = EventState.LOST; change(event, time, "lost", "not_observed_in_complete_frame")
            } else null
        }
    }
    fun finish(time: Double): List<EventChange> = liveEvents().map { change(it, time, "censored", "input_ended_without_resolution") }
}

fun bboxIou(a: List<Double>, b: List<Double>): Double {
    val intersection = maxOf(0.0, minOf(a[2], b[2]) - maxOf(a[0], b[0])) * maxOf(0.0, minOf(a[3], b[3]) - maxOf(a[1], b[1]))
    val union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return if (union > 0) intersection / union else 0.0
}

class PerceptionMonitor(private val config: AlertConfig, private val streamId: String) {
    var current = PerceptionStatus(); private set
    var alerted = false
    private var nextId = 1
    fun update(time: Double, status: String, reason: String?): List<EventChange> {
        var previous = current; val changes = mutableListOf<EventChange>()
        if (status == "available") {
            previous.eventId?.let { changes += EventChange(streamId, it, time, "perception", "resolved", EventState.RESOLVED, "perception_recovered") }
            current = PerceptionStatus(); alerted = false; return changes
        }
        if (previous.eventId == null) {
            val id = "H%06d".format(nextId++)
            previous = PerceptionStatus(status, id, EventState.CANDIDATE, time, reason); alerted = false
            changes += EventChange(streamId, id, time, "perception", "created", EventState.CANDIDATE, reason ?: status)
        }
        var state = previous.state
        if (state == EventState.CANDIDATE && time - previous.sinceMs!! >= config.perceptionConfirmMs) {
            state = EventState.ACTIVE
            changes += EventChange(streamId, previous.eventId!!, time, "perception", "activated", state, "persistent_perception_limitation")
        } else if (previous.status != status) changes += EventChange(streamId, previous.eventId!!, time, "perception", "updated", state!!, reason ?: status)
        current = PerceptionStatus(status, previous.eventId, state, previous.sinceMs, reason)
        return changes
    }
    fun finish(time: Double): List<EventChange> = current.eventId?.let {
        listOf(EventChange(streamId, it, time, "perception", "censored", current.state!!, "input_ended_with_perception_limitation"))
    } ?: emptyList()
}

class AlertScheduler(private val config: AlertConfig, private val streamId: String) {
    private data class History(val lastMs: Double, val highestLevel: RiskLevel, val highestScore: Double)
    private val history = mutableMapOf<String, History>()
    private var nextId = 1
    fun forget(id: String) { history.remove(id) }
    /** PC replay commits immediately; the Android presenter commits only after accepting playback. */
    fun schedule(events: List<EventSnapshot>, monitor: PerceptionMonitor, time: Double, autoCommit: Boolean = true): List<AlertEvent> {
        val eligible = events.mapNotNull { event ->
            val risk = event.assessment ?: return@mapNotNull null
            if (event.state != EventState.ACTIVE || !event.observedNow || risk.level.ordinal < RiskLevel.valueOf(config.minimumLevel).ordinal) return@mapNotNull null
            val previous = history[event.eventId]
            val type = when {
                previous == null -> "first"
                risk.level.ordinal > previous.highestLevel.ordinal && risk.score >= previous.highestScore + config.escalationMinScoreDelta && time - previous.lastMs >= config.escalationIntervalMs -> "escalation"
                config.repeatEnabled && time - previous.lastMs >= config.ordinaryIntervalMs -> "repeat"
                else -> null
            }
            type?.let { event to it }
        }.sortedWith(compareByDescending<Pair<EventSnapshot, String>> { it.first.assessment!!.level.ordinal }
            .thenByDescending { it.first.assessment!!.spatialScore }.thenByDescending { it.first.assessment!!.score }.thenBy { it.first.eventId })
        val alerts = mutableListOf<AlertEvent>()
        var proposedId = nextId
        val health = monitor.current
        if (health.state == EventState.ACTIVE && !monitor.alerted) {
            alerts += AlertEvent(streamId, "A%06d".format(proposedId++), health.eventId!!, time, null, null,
                "perception_limited", "${health.status}: ${health.reason}; environment unknown", emptyList())
        }
        for ((event, type) in eligible) {
            if (alerts.size >= config.maxAlertsPerFrame) break
            val risk = event.assessment!!
            var reason = risk.reasons.joinToString("; ")
            if (risk.quality.score < config.limitedQualityThreshold) reason += "; perception quality limited; risk is not reduced by quality"
            alerts += AlertEvent(streamId, "A%06d".format(proposedId++), event.eventId, time, risk.level, risk.score,
                type, reason, event.sourceTrackIds, risk.quality.score)
        }
        if (autoCommit) alerts.forEach { accept(it, monitor) }
        return alerts
    }
    fun accept(alert: AlertEvent, monitor: PerceptionMonitor) {
        nextId++
        if (alert.alertType == "perception_limited") { monitor.alerted = true; return }
        val level = alert.riskLevel ?: return; val score = alert.riskScore ?: return
        val previous = history[alert.eventId]
        history[alert.eventId] = History(alert.timestampMs,
            if (previous == null || level.ordinal > previous.highestLevel.ordinal) level else previous.highestLevel,
            maxOf(score, previous?.highestScore ?: score))
    }
}

/** Ported orchestration with independent state per input session and monotonic source time. */
class AssessmentEngine(val config: RiskConfig = RiskConfig(), val streamId: String = "video") {
    val features = FeatureExtractor(config.features)
    val scorer = RiskScorer(config.scoring)
    val manager = EventManager(config.association, streamId)
    val monitor = PerceptionMonitor(config.alerts, streamId)
    val scheduler = AlertScheduler(config.alerts, streamId)
    private var lastIndex = -1; private var lastTimestamp: Double? = null; private var finished = false
    fun step(frame: PFrame, autoCommitAlerts: Boolean = true): StepResult {
        require(!finished) { "Assessment stream finished" }
        require(frame.frameIndex == lastIndex + 1) { "Frames must start at zero and be contiguous" }
        require(lastTimestamp == null || frame.timestampMs >= lastTimestamp!!) { "Timestamp must not decrease" }
        val warnings = mutableListOf<String>()
        lastTimestamp?.let {
            val interval = frame.timestampMs - it
            if (interval == 0.0) warnings += "duplicate_timestamp_motion_unavailable"
            else if (interval > config.alerts.maxFrameGapMs) warnings += "frame_gap_exceeds_limit"
        }
        val geometry = mutableMapOf<Int, SpatialFeatures>(); val invalid = mutableMapOf<Int, String>()
        frame.detections.forEachIndexed { index, d ->
            try { geometry[index] = spatialFeatures(d, frame.frameWidth, frame.frameHeight, config.features) }
            catch (error: IllegalArgumentException) { invalid[index] = error.message ?: "invalid_bbox"; warnings += "detection_$index: ${invalid[index]}" }
        }
        val complete = frame.status == "ok" && invalid.isEmpty()
        val changes = manager.prepare(frame.timestampMs, complete).toMutableList()
        changes.forEach { features.forget(it.eventId); scorer.forget(it.eventId); scheduler.forget(it.eventId) }
        features.beginFrame(frame, complete)
        val observations = invalid.toSortedMap().map { (index, error) ->
            val d = frame.detections[index]
            RiskObservation(index, d.trackId, d.classId, d.className, null, null, null, null, error)
        }.toMutableList()
        manager.associate(frame, geometry).forEach { association ->
            val index = association.detectionIndex; val event = association.event; val d = frame.detections[index]
            val feature = features.observe(event.eventId, d, frame, geometry.getValue(index))
            val assessment = scorer.assess(event.eventId, feature, frame.timestampMs)
            changes += manager.observe(association, d, geometry.getValue(index), assessment, frame.timestampMs)
            observations += RiskObservation(index, d.trackId, d.classId, d.className, event.eventId, association.method, feature, assessment)
        }
        changes += manager.endFrame(frame.timestampMs, complete)
        manager.liveEvents().filter { !it.observedNow }.forEach { scorer.interrupt(it.eventId) }
        val (status, reason) = when {
            frame.status == "processing_error" -> "unavailable" to frame.error
            warnings.isNotEmpty() -> "limited" to warnings.joinToString("; ")
            observations.any { it.assessment != null && it.assessment.quality.score < config.alerts.limitedQualityThreshold } -> "limited" to "one_or_more_targets_have_limited_perception_quality"
            else -> "available" to null
        }
        changes += monitor.update(frame.timestampMs, status, reason)
        val snapshots = manager.liveEvents().map { it.snapshot() }
        val alerts = scheduler.schedule(snapshots, monitor, frame.timestampMs, autoCommitAlerts)
        lastIndex = frame.frameIndex; lastTimestamp = frame.timestampMs
        return StepResult(RiskFrame(streamId, frame.frameIndex, frame.timestampMs, frame.timestampSource, frame.status,
            monitor.current, observations.sortedBy { it.detectionIndex }, snapshots, warnings), changes, alerts)
    }
    fun acceptAlert(alert: AlertEvent) { scheduler.accept(alert, monitor) }
    fun finish(): List<EventChange> {
        if (finished) return emptyList()
        finished = true
        return lastTimestamp?.let { manager.finish(it) + monitor.finish(it) } ?: emptyList()
    }
}
