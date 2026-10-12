package org.sightguard.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.nio.FloatBuffer
import java.security.MessageDigest
import org.json.JSONObject
import kotlin.math.roundToInt

/** One CPU session, accessed only by its owning serial inference executor. */
class Detector(context: Context, val experimental: Boolean = !context.getSharedPreferences("perception", Context.MODE_PRIVATE).getBoolean("baseline_detector", false)) : AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    val modelSha: String
    private val classMapping: List<Int>

    init {
        val bytes = context.assets.open(if(experimental) "yoloe26n.onnx" else "yolo26n.onnx").use { it.readBytes() }
        modelSha = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
        val manifest = JSONObject(context.assets.open(if(experimental) "yoloe_model_manifest.json" else "model_manifest.json").bufferedReader().use { it.readText() })
        require(modelSha == manifest.getString("onnx_sha256")) { "模型 SHA 与清单不一致" }
        classMapping = if(experimental) {
            require(manifest.getString("schema_version")=="sightguard.yoloe26.v1")
            val mapping=manifest.getJSONArray("app_class_ids")
            List(mapping.length()) {mapping.getInt(it)}.also {
                require(it==DetectorTaxonomy.yoloeMapping && manifest.getJSONArray("names").length()==90)
            }
        } else (0..79).toList()
        OrtSession.SessionOptions().use { options ->
            options.setIntraOpNumThreads(2)
            options.setInterOpNumThreads(1)
            session = environment.createSession(bytes, options)
        }
        try {
            inputName = session.inputNames.single()
            val input = session.inputInfo.getValue(inputName).info as TensorInfo
            val output = session.outputInfo.values.single().info as TensorInfo
            require(input.shape.contentEquals(longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong()))) { "不支持的模型输入: ${input.shape.contentToString()}" }
            // This is the verified YOLO26 end-to-end export, not a YOLOv8 feature map.
            require(output.shape.size == 3 && output.shape[0] == 1L && output.shape[2] == 6L) { "不支持的模型输出: ${output.shape.contentToString()}" }
            android.util.Log.i("SG_MODEL", "loaded kind=${if(experimental) "YOLOE26n" else "YOLO26n"} sha=$modelSha input=${input.shape.contentToString()} output=${output.shape.contentToString()}")
        } catch (error: Exception) { session.close(); throw error }
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        val ratio = minOf(SIZE.toFloat() / bitmap.width, SIZE.toFloat() / bitmap.height)
        val scaledWidth = (bitmap.width * ratio).roundToInt().coerceAtLeast(1)
        val scaledHeight = (bitmap.height * ratio).roundToInt().coerceAtLeast(1)
        val padX = (SIZE - scaledWidth) / 2
        val padY = (SIZE - scaledHeight) / 2
        val scaled = Bitmap.createScaledBitmap(bitmap, scaledWidth, scaledHeight, true)
        val input = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
        Canvas(input).apply {
            drawColor(Color.rgb(114, 114, 114))
            drawBitmap(scaled, padX.toFloat(), padY.toFloat(), Paint())
        }
        if (scaled !== bitmap) scaled.recycle()
        val pixels = IntArray(SIZE * SIZE)
        input.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        input.recycle()
        val values = FloatArray(3 * SIZE * SIZE)
        pixels.forEachIndexed { index, pixel ->
            values[index] = Color.red(pixel) / 255f
            values[index + pixels.size] = Color.green(pixel) / 255f
            values[index + 2 * pixels.size] = Color.blue(pixel) / 255f
        }
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(values), longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val rows = (result[0].value as Array<Array<FloatArray>>)[0]
                // Top-K detections are already selected by the end-to-end model; no extra NMS.
                return rows.mapNotNull { row ->
                    if (row.size != 6 || row.any { !it.isFinite() } || row[4] !in MIN_SCORE..1f) return@mapNotNull null
                    val classId = row[5].roundToInt()
                    if (classId !in classMapping.indices || kotlin.math.abs(row[5]-classId)>0.001f) return@mapNotNull null
                    val left = ((row[0] - padX) / scaledWidth).coerceIn(0f, 1f)
                    val top = ((row[1] - padY) / scaledHeight).coerceIn(0f, 1f)
                    val right = ((row[2] - padX) / scaledWidth).coerceIn(0f, 1f)
                    val bottom = ((row[3] - padY) / scaledHeight).coerceIn(0f, 1f)
                    if (right <= left || bottom <= top) null else Detection(classMapping[classId], row[4], left, top, right, bottom, if(experimental) classId else null)
                }
            }
        }
    }

    internal fun inferTensor(values: FloatArray): Array<FloatArray> {
        require(values.size==3*SIZE*SIZE && values.all {it.isFinite()})
        OnnxTensor.createTensor(environment,FloatBuffer.wrap(values),longArrayOf(1,3,SIZE.toLong(),SIZE.toLong())).use {input ->
            session.run(mapOf(inputName to input)).use {output ->
                @Suppress("UNCHECKED_CAST")
                return (output[0].value as Array<Array<FloatArray>>)[0]
            }
        }
    }
    override fun close() { session.close() }
    companion object { const val SIZE = 320; const val MIN_SCORE = 0.35f }
}
