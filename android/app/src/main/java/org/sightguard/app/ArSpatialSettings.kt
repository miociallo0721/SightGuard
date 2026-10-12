package org.sightguard.app

import android.content.Context
import org.json.JSONObject

fun loadArSpeechConfig(context: Context): ArSpeechConfig {
    val j=JSONObject(context.assets.open("ar_spatial_config.json").bufferedReader().use {it.readText()})
    require(j.getString("schema_version")=="sightguard.ar-spatial.v1" && !j.getBoolean("numeric_distance_speech") && !j.getBoolean("navigation_instructions"))
    return ArSpeechConfig(j.getLong("max_source_age_ms"),j.getLong("max_raw_depth_age_ms"),j.getLong("max_pair_ms"),j.getInt("confirmations"),
        j.getInt("minimum_points"),j.getDouble("minimum_confidence").toFloat(),j.getDouble("plane_residual_m").toFloat(),
        j.getDouble("minimum_inlier_fraction").toFloat(),j.getDouble("near_m").toFloat(),j.getDouble("approach_delta_m").toFloat(),j.getLong("cooldown_ms"),j.getLong("recovery_ms"))
}
