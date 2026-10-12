package org.sightguard.app

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

fun loadLocalSpaceConfig(context: Context): LocalSpaceConfig {
    val j=JSONObject(context.assets.open("local_space_config.json").bufferedReader().use {it.readText()})
    require(j.getInt("schema_version")==1)
    return LocalSpaceConfig(j.getLong("max_age_ms"),j.getLong("max_gap_ms"),j.getInt("confirmations"),j.getLong("cooldown_ms"),
        j.getDouble("min_ground").toFloat(),j.getDouble("max_occupancy").toFloat(),j.getDouble("min_connected").toFloat(),j.getDouble("tie_margin").toFloat())
}
fun localSpaceJson(r: LocalSpaceRecord): JSONObject = JSONObject().put("schema_version",1)
    .put("source_frame_key","${r.key.generation}:${r.key.epoch}:${r.key.frameIndex}").put("source_ms",r.key.timestampMs)
    .put("status",r.status).put("reason",r.reason).put("scope","IMAGE_CANDIDATE_ONLY")
    .put("direction",r.direction?.name ?: JSONObject.NULL).put("support",r.support).put("deadline_ms",r.validUntilMs)
    .put("event_id",r.eventId ?: JSONObject.NULL).put("text",r.text ?: JSONObject.NULL)
    .put("bands",JSONArray(r.bands.map {JSONObject().put("direction",it.direction.name).put("connected_fraction",it.connectedFraction)
        .put("reaches_top",it.reachesTop).put("unknown_fraction",it.unknownFraction).put("eligible",it.eligible)}))
    .put("physical_width_m",JSONObject.NULL).put("body_alignment","UNKNOWN").put("risk_score_changed",false)
