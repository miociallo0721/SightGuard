package org.sightguard.app

import android.content.Context
import android.graphics.Bitmap
import org.json.JSONObject

fun loadEnvironmentConfig(context: Context): EnvironmentConfig {
    val json=JSONObject(context.assets.open("semantic_environment_config.json").bufferedReader().use { it.readText() })
    require(!json.getBoolean("approach_enabled")) { "Approach evidence has not been validated" }
    val roi=json.getJSONArray("roi")
    val structures=json.optJSONObject("structures")
    val thresholds=EnvironmentConfig().thresholds.toMutableMap()
    EnvironmentKind.entries.forEach { kind -> structures?.optJSONObject(kind.name)?.let { value ->
        thresholds[kind]=StructureThreshold(value.getDouble("min_roi_fraction").toFloat(),value.getInt("min_cells"),value.getDouble("min_confidence").toFloat(),value.getInt("confirmation_count"))
    } }
    return EnvironmentConfig(json.getInt("schema_version"),json.getLong("max_source_age_ms"),json.getLong("sample_interval_ms"),
        json.getLong("recovery_ms"),json.getLong("cooldown_ms"),json.getInt("confirmation_count"),json.getLong("min_confirmation_span_ms"),
        json.getDouble("min_confidence").toFloat(),json.getDouble("min_roi_fraction").toFloat(),json.getInt("min_cells"),
        roi.getDouble(0).toFloat(),roi.getDouble(1).toFloat(),roi.getDouble(2).toFloat(),roi.getDouble(3).toFloat(),json.getDouble("max_scene_difference").toFloat(),thresholds,
        json.getJSONArray("automatic_kinds").let { a -> (0 until a.length()).map {EnvironmentKind.valueOf(a.getString(it))}.toSet() },
        json.optBoolean("furniture_family_confidence",false),json.optInt("preferred_semantic_input_size",256),
        json.optJSONObject("automatic_model_requirements")?.let { requirements -> requirements.keys().asSequence().associate {EnvironmentKind.valueOf(it) to requirements.getString(it)} } ?: emptyMap(),
        json.optLong("resolution_fallback_age_ms",400),json.optInt("resolution_fallback_count",2),json.optInt("resolution_fallback_primary_slow",5))
}

fun semanticSceneSignature(bitmap: Bitmap): FloatArray {
    val small=Bitmap.createScaledBitmap(bitmap,16,16,true)
    val pixels=IntArray(256); small.getPixels(pixels,0,16,0,0,16,16)
    if(small!==bitmap) small.recycle()
    return FloatArray(768) { i -> ((pixels[i/3] shr (16-8*(i%3))) and 255)/255f }
}

fun loadWallResearchConfig(context: Context): WallResearchConfig {
    val j=JSONObject(context.assets.open("wall_research_config.json").bufferedReader().use {it.readText()})
    require(j.getInt("schema_version")==1 && !j.getBoolean("automatic_alert_enabled")) {"Unvalidated wall approach must stay shadow-only"}
    return WallResearchConfig(maxSourceAgeMs=j.getLong("max_source_age_ms"),minPairMs=j.getDouble("min_pair_ms"),maxPairMs=j.getDouble("max_pair_ms"),
        maxFeatures=j.getInt("max_features"),minFeatures=j.getInt("min_features"),featureScore=j.getDouble("feature_score").toFloat(),
        maxPatchResidual=j.getDouble("max_patch_residual").toFloat(),maxFitResidual=j.getDouble("max_fit_residual").toFloat(),
        minInlierFraction=j.getDouble("min_inlier_fraction").toFloat(),maxGyro=j.getDouble("max_gyro_rad_s").toFloat(),maxRotationRad=j.getDouble("max_rotation_rad").toFloat(),
        minExpansion=j.getDouble("min_expansion").toFloat(),maxExpansion=j.getDouble("max_expansion").toFloat(),confirmationWindows=j.getInt("confirmation_windows"),maxFlowMs=j.getLong("max_flow_ms"))
}
