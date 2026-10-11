package org.sightguard.app

import org.sightguard.app.core.bboxIou
import kotlin.math.hypot

/** Image-plane association only: no ReID, BoT-SORT, camera-motion or world-speed claim. */
data class TrackerConfig(
    val minimumConfidence: Float = .35f,
    val recoveryMs: Long = 2500,
    val predictionLimitMs: Long = 500,
    val maximumCenterDistance: Double = .18,
    val minimumIou: Double = .10,
    val ambiguityMargin: Double = .08,
    val duplicateIou: Double = .80,
    val duplicateCenterDistance: Double = .03,
)
data class TrackedDetection(val detection: Detection, val trackId: Long)

class LightTracker(private val config: TrackerConfig = TrackerConfig()) {
    private data class Track(val id: Long, var detection: Detection, var lastMs: Long, var vx: Double = 0.0, var vy: Double = 0.0)
    private data class Edge(val index: Int, val id: Long, val score: Double)
    private val tracks = linkedMapOf<Long, Track>()
    private var nextId = 1L
    private var previousMs: Long? = null

    fun reset() { tracks.clear(); nextId = 1; previousMs = null }
    fun update(raw: List<Detection>, timeMs: Long): List<TrackedDetection> {
        require(previousMs == null || timeMs >= previousMs!!) { "Tracker source time decreased; reset on seek" }
        previousMs = timeMs
        tracks.entries.removeAll { timeMs - it.value.lastMs > config.recoveryMs }
        // End-to-end YOLO can produce coincident rows. Preserve raw boxes in the HUD,
        // but do not let near-identical rows manufacture multiple risk identities.
        val detections = mutableListOf<Detection>()
        raw.filter { it.confidence >= config.minimumConfidence }.sortedByDescending { it.confidence }.forEach { d ->
            if (detections.none { it.classId == d.classId && distance(it, d) <= config.duplicateCenterDistance && bboxIou(box(it), box(d)) >= config.duplicateIou }) detections += d
        }
        val edges = mutableListOf<Edge>()
        detections.forEachIndexed { index, d -> tracks.values.forEach { track ->
            if (track.detection.classId != d.classId) return@forEach
            val dt = (timeMs - track.lastMs).coerceAtMost(config.predictionLimitMs) / 1000.0
            val predicted = box(track.detection).mapIndexed { axis, value -> value + (if (axis % 2 == 0) track.vx else track.vy) * dt }
            val actual = box(d)
            val centerDistance = hypot((actual[0] + actual[2] - predicted[0] - predicted[2]) / 2, (actual[1] + actual[3] - predicted[1] - predicted[3]) / 2)
            val overlap = bboxIou(predicted, actual)
            if (centerDistance <= config.maximumCenterDistance && overlap >= config.minimumIou) {
                edges += Edge(index, track.id, .6 * overlap + .4 * (1 - centerDistance / config.maximumCenterDistance))
            }
        } }
        val ordered = edges.sortedWith(compareByDescending<Edge> { it.score }.thenBy { it.id }.thenBy { it.index })
        val byIndex = ordered.groupBy { it.index }; val byId = ordered.groupBy { it.id }
        val matches = mutableMapOf<Int, Long>()
        ordered.forEach { edge ->
            val a = byIndex.getValue(edge.index); val b = byId.getValue(edge.id)
            if (edge == a.first() && edge == b.first() && listOf(a, b).none { it.size > 1 && edge.score - it[1].score <= config.ambiguityMargin }) matches[edge.index] = edge.id
        }
        return detections.mapIndexed { index, d ->
            val id = matches[index]
            val track = id?.let(tracks::get)
            if (track == null) {
                val created = Track(nextId++, d, timeMs); tracks[created.id] = created
                TrackedDetection(d, created.id)
            } else {
                val dt = (timeMs - track.lastMs) / 1000.0
                if (dt > 0 && timeMs - track.lastMs <= config.predictionLimitMs) {
                    val vx = ((d.left + d.right) - (track.detection.left + track.detection.right)) / (2 * dt)
                    val vy = ((d.top + d.bottom) - (track.detection.top + track.detection.bottom)) / (2 * dt)
                    track.vx = .5 * track.vx + .5 * vx; track.vy = .5 * track.vy + .5 * vy
                } else { track.vx = 0.0; track.vy = 0.0 }
                track.detection = d; track.lastMs = timeMs
                TrackedDetection(d, track.id)
            }
        }
    }
    private fun box(d: Detection) = listOf(d.left.toDouble(), d.top.toDouble(), d.right.toDouble(), d.bottom.toDouble())
    private fun distance(a: Detection, b: Detection) = hypot(((a.left + a.right) - (b.left + b.right)).toDouble() / 2, ((a.top + a.bottom) - (b.top + b.bottom)).toDouble() / 2)
}
