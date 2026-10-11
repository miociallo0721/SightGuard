package org.sightguard.app

import android.os.SystemClock
import org.json.JSONArray
import org.json.JSONObject

/** Bounded metadata only. No camera pixels or unbounded per-frame log retained. */
class AwarenessDiagnostics(private val maximum: Int=1500) {
    private val records=ArrayDeque<JSONObject>()
    @Synchronized fun record(type: String, values: JSONObject) {
        records.addLast(JSONObject().put("type",type).put("elapsed_ms",SystemClock.elapsedRealtime()).put("data",values))
        while(records.size>maximum) records.removeFirst()
    }
    @Synchronized fun snapshot(session: JSONObject): String = JSONObject()
        .put("schema_version",1).put("app_version","0.4-accessible-awareness")
        .put("device",android.os.Build.MODEL).put("api",android.os.Build.VERSION.SDK_INT)
        .put("semantic_model_sha256",SpatialDetector.MODEL_SHA).put("session",session)
        .put("wall_proximity_automatic",false).put("records",JSONArray(records.toList())).toString(2)
}

fun environmentSupportJson(support: EnvironmentSupport): JSONObject = JSONObject()
    .put("kind",support.kind.name).put("raw_class_ids",JSONArray(support.rawClassIds.sorted()))
    .put("roi_fraction",support.roiFraction).put("confidence",support.confidence).put("confidence_p10",support.confidenceP10)
    .put("cells",support.cells).put("supported",support.supported).put("direction",support.direction.name)
    .put("bounds",JSONArray(listOf(support.left,support.top,support.right,support.bottom)))
