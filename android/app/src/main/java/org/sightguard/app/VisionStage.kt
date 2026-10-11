package org.sightguard.app

import android.content.Context
import android.graphics.Bitmap
import org.sightguard.app.core.*

data class VisionResult(val raw: List<Detection>, val records: PFrame, val risk: StepResult)

/** A single serially-owned stage shared by CameraX and decoded local video. */
class VisionStage(context: Context, private val source: String) : AutoCloseable {
    private val detector = Detector(context)
    val modelSha get() = detector.modelSha
    private var tracker = LightTracker()
    private var engine = newEngine()
    private var frameIndex = 0
    val presenter = AlertPresenter()
    private fun newEngine() = AssessmentEngine(RiskConfig().let { it.copy(
        association = it.association.copy(recoveryWindowMs = 2500.0),
        alerts = it.alerts.copy(maxAlertsPerFrame = 8),
    ) }, source)
    fun reset() { engine.finish(); engine = newEngine(); tracker.reset(); presenter.reset(); frameIndex = 0 }
    fun process(bitmap: Bitmap, timestampMs: Long, timestampSource: String): VisionResult {
        val raw = detector.detect(bitmap)
        val tracked = tracker.update(raw, timestampMs)
        val detections = tracked.map { t -> val d = t.detection; PDetection(t.trackId, d.classId, Labels.coco80[d.classId], d.confidence.toDouble(),
            listOf(d.left * bitmap.width.toDouble(), d.top * bitmap.height.toDouble(), d.right * bitmap.width.toDouble(), d.bottom * bitmap.height.toDouble())) }
        val frame = PFrame(frameIndex++, timestampMs.toDouble(), bitmap.width, bitmap.height, detections, timestampSource = timestampSource)
        return VisionResult(raw, frame, engine.step(frame, autoCommitAlerts = false))
    }
    fun accept(alert: PresentedAlert) { (alert.merged + alert.alert).forEach(engine::acceptAlert) }
    override fun close() { engine.finish(); tracker.reset(); presenter.reset(); detector.close() }
}
