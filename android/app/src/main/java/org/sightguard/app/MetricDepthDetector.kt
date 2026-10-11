package org.sightguard.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import java.security.MessageDigest

data class MetricResearchSettings(val intervalMs: Long,val minimumMemoryMb: Long,val wall: WallMetricConfig) {
    init {require(intervalMs>=1000 && minimumMemoryMb>=256)}
}

fun loadMetricResearchSettings(context: Context): MetricResearchSettings {
    val root=JSONObject(context.assets.open("metric_research_config.json").bufferedReader().use {it.readText()})
    require(root.getString("schema_version")=="sightguard.metric-research-config.v1" &&
        root.getBoolean("shadow_only") && !root.getBoolean("default_enabled") && !root.getBoolean("automatic_speech"))
    val wall=root.getJSONObject("wall")
    return MetricResearchSettings(root.getLong("sample_interval_ms"),root.getLong("minimum_available_memory_mb"),
        WallMetricConfig(wall.getDouble("minimum_confidence").toFloat(),wall.getInt("minimum_component_pixels"),
            wall.getDouble("minimum_roi_coverage").toFloat(),wall.getDouble("minimum_valid_fraction").toFloat(),
            wall.getLong("maximum_source_age_ms")))
}

/** Optional research adapter. One CPU ORT graph, shared immutable weights, no risk/audio calls. */
class MetricDepthDetector(context: Context): AutoCloseable {
    private val application=context.applicationContext
    private val environment=OrtEnvironment.getEnvironment()
    private val manifest=JSONObject(application.assets.open("metric_research/metric_manifest.json").bufferedReader().use {it.readText()})
    private var session: OrtSession?=null
    private var sessionShape: MetricShape?=null
    private var pixels=IntArray(0)
    private var tensorValues=FloatArray(0)
    private var graphSha=""
    val weightsSha: String=manifest.getJSONObject("shared_weights").getString("sha256")
    private val directory=File(application.noBackupFilesDir,"metric_research/${weightsSha.take(16)}")

    init {
        require(manifest.getString("schema_version")=="sightguard.metric-depth-model.v1")
        require(manifest.getBoolean("shadow_only") && !manifest.getBoolean("default_enabled") && !manifest.getBoolean("automatic_speech"))
        require(manifest.getString("quantity")=="predicted_optical_axis_z_m" && manifest.getInt("max_depth_m")==20)
        require(manifest.getString("checkpoint_sha256")=="b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545")
    }

    private fun hash(file: File): String {
        val digest=MessageDigest.getInstance("SHA-256")
        file.inputStream().use {input->val buffer=ByteArray(65536)
            while(true) {val count=input.read(buffer);if(count<0) break;digest.update(buffer,0,count)}}
        return digest.digest().joinToString("") {"%02x".format(it)}
    }

    private fun materialize(name: String, expected: String): File {
        require(name.matches(Regex("[a-zA-Z0-9_.]+")))
        directory.mkdirs()
        val target=File(directory,name)
        if(target.isFile && hash(target)==expected) return target
        val pending=File(directory,"$name.pending")
        try {
            application.assets.open("metric_research/$name").use {source->pending.outputStream().use {source.copyTo(it,65536)}}
            require(hash(pending)==expected) {"Metric model SHA mismatch: $name"}
            check(pending.renameTo(target)) {"Metric model extraction failed"}
        } finally {pending.delete()}
        return target
    }

    private fun ensureSession(shape: MetricShape): OrtSession {
        session?.let {if(sessionShape==shape) return it}
        session?.close();session=null;sessionShape=null
        val graphs=manifest.getJSONArray("graphs")
        val graph=(0 until graphs.length()).map {graphs.getJSONObject(it)}.firstOrNull {
            val input=it.getJSONArray("input_shape");input.getInt(2)==shape.height && input.getInt(3)==shape.width
        } ?: error("Unsupported metric shape: $shape")
        val started=SystemClock.elapsedRealtime()
        materialize("metric_weights.bin",weightsSha)
        val path=materialize(graph.getString("file"),graph.getString("sha256"))
        val loaded=OrtSession.SessionOptions().use {options->
            options.setIntraOpNumThreads(1);options.setInterOpNumThreads(1)
            environment.createSession(path.absolutePath,options)
        }
        try {
            require((loaded.inputInfo.getValue("pixel_values").info as TensorInfo).shape.contentEquals(
                longArrayOf(1,3,shape.height.toLong(),shape.width.toLong())))
            require((loaded.outputInfo.getValue("predicted_metric_z_m").info as TensorInfo).shape.contentEquals(
                longArrayOf(1,shape.height.toLong(),shape.width.toLong())))
        } catch(error: Exception) {loaded.close();throw error}
        session=loaded;sessionShape=shape;graphSha=graph.getString("sha256")
        Log.i("SG_METRIC","loadMs=${SystemClock.elapsedRealtime()-started} shape=$shape graphSha=$graphSha weightsSha=$weightsSha pssKb=${android.os.Debug.getPss()} shadow=true")
        return loaded
    }

    fun prepare(bitmap: Bitmap, shape: MetricShape): FloatArray {
        require(MetricDepthMath.shape(bitmap.width,bitmap.height)==shape) {"Metric input geometry unsupported"}
        if(pixels.size!=bitmap.width*bitmap.height) pixels=IntArray(bitmap.width*bitmap.height)
        if(tensorValues.size!=3*shape.width*shape.height) tensorValues=FloatArray(3*shape.width*shape.height)
        bitmap.getPixels(pixels,0,bitmap.width,0,0,bitmap.width,bitmap.height)
        return MetricDepthMath.prepare(pixels,bitmap.width,bitmap.height,shape,tensorValues)
    }

    fun infer(values: FloatArray, shape: MetricShape): FloatArray {
        require(values.size==3*shape.width*shape.height && shape in MetricDepthMath.supportedShapes)
        val loaded=ensureSession(shape)
        OnnxTensor.createTensor(environment,FloatBuffer.wrap(values),
            longArrayOf(1,3,shape.height.toLong(),shape.width.toLong())).use {input->
            loaded.run(mapOf("pixel_values" to input)).use {output->
                @Suppress("UNCHECKED_CAST")
                val rows=(output[0].value as Array<Array<FloatArray>>)[0]
                return FloatArray(shape.width*shape.height) {index->rows[index/shape.width][index%shape.width]}
            }
        }
    }

    fun analyze(bitmap: Bitmap,key: SpatialFrameKey): MetricDepthEvidence {
        require(bitmap.width==key.width && bitmap.height==key.height)
        val started=SystemClock.elapsedRealtime()
        val shape=MetricDepthMath.shape(bitmap.width,bitmap.height) ?: error("unsupported_aspect_ratio")
        val values=infer(prepare(bitmap,shape),shape)
        val completed=SystemClock.elapsedRealtime()
        Log.i("SG_METRIC","frame=${key.frameIndex} clock=${key.timestampClock} sourceAgeMs=${completed-key.capturedAtMs} totalMs=${completed-started} shape=$shape pssKb=${android.os.Debug.getPss()} automatic=false")
        return MetricDepthEvidence(key,values,shape,completed-started,completed,graphSha)
    }

    override fun close() {session?.close();session=null;sessionShape=null;pixels=IntArray(0);tensorValues=FloatArray(0)}
}

fun metricWallJson(record: WallMetricRecord): JSONObject = JSONObject()
    .put("schema_version","sightguard.wall-metric.v1").put("state",record.status)
    .put("frame",record.key.frameIndex).put("generation",record.key.generation).put("epoch",record.key.epoch)
    .put("timestamp_ms",record.key.timestampMs).put("captured_at_ms",record.key.capturedAtMs)
    .put("generated_at_elapsed_ms",record.generatedAtElapsedMs).put("source_age_basis",record.sourceAgeBasis)
    .put("capture_timestamp_ns",record.key.captureTimestampNs ?: JSONObject.NULL)
    .put("width",record.key.width).put("height",record.key.height).put("rotation_degrees",record.key.rotationDegrees)
    .put("timestamp_clock",record.key.timestampClock).put("source_age_ms",record.sourceAgeMs)
    .put("wall_present",record.wallPresent ?: JSONObject.NULL)
    .put("predicted_metric_depth",record.predictedMetricDepth ?: JSONObject.NULL)
    .put("median_m",record.median ?: JSONObject.NULL).put("p10_m",record.p10 ?: JSONObject.NULL).put("p90_m",record.p90 ?: JSONObject.NULL)
    .put("valid_pixel_fraction",record.validPixelFraction).put("wall_coverage",record.wallCoverage)
    .put("semantic_confidence",record.semanticConfidence ?: JSONObject.NULL)
    .put("component_count",record.componentCount).put("wall_pixels",record.wallPixels)
    .put("model_latency_ms",record.modelLatencyMs ?: JSONObject.NULL).put("model_sha256",record.modelSha)
    .put("motion_quality",record.motionQuality).put("motion_sync",record.motionSync)
    .put("quantity","predicted_optical_axis_z_m_not_euclidean_range")
    .put("uncertainty","P10_P90_are_spatial_not_accuracy_bounds_no_phone_tape_validation")
    .put("rejection_reasons",org.json.JSONArray(record.rejectionReasons)).put("automatic_alert_enabled",false)
