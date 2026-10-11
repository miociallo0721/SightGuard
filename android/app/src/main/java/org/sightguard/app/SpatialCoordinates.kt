package org.sightguard.app

import kotlin.math.roundToInt

/** Upright source image to a mean-padded square. Coordinates are normalized source coordinates. */
data class SquareTransform(val sourceWidth: Int, val sourceHeight: Int, val size: Int) {
    init { require(sourceWidth > 0 && sourceHeight > 0 && size > 0) }
    private val scale = minOf(size.toFloat() / sourceWidth, size.toFloat() / sourceHeight)
    val resizedWidth = (sourceWidth * scale).roundToInt().coerceIn(1, size)
    val resizedHeight = (sourceHeight * scale).roundToInt().coerceIn(1, size)
    val padLeft = (size - resizedWidth) / 2
    val padTop = (size - resizedHeight) / 2
    fun sourceX(modelX: Float) = (modelX - padLeft) / resizedWidth
    fun sourceY(modelY: Float) = (modelY - padTop) / resizedHeight
    fun modelX(sourceX: Float) = padLeft + sourceX * resizedWidth
    fun modelY(sourceY: Float) = padTop + sourceY * resizedHeight
}

data class DepthStatistics(val validFraction: Float, val median: Float?, val q10: Float?, val q90: Float?, val framePercentile: Float?, val relation: String)

object RelativeDepthRules {
    const val MIN_VALID_FRACTION = .8f
    const val MIN_RELATIVE_SPREAD = .01f
    const val NEAR_PERCENTILE = .7f
    const val FAR_PERCENTILE = .3f
    fun quantile(sorted: List<Float>, q: Float): Float {
        require(sorted.isNotEmpty())
        val position = q.coerceIn(0f, 1f) * (sorted.size - 1)
        val left = position.toInt(); val right = (left + 1).coerceAtMost(sorted.lastIndex)
        return sorted[left] + (position - left) * (sorted[right] - sorted[left])
    }
    fun summarize(values: FloatArray, frameValues: FloatArray): DepthStatistics {
        val valid = values.filter { it.isFinite() && it >= 0 }.sorted()
        val frame = frameValues.filter { it.isFinite() && it >= 0 }.sorted()
        val fraction = if (values.isEmpty()) 0f else valid.size.toFloat() / values.size
        if (valid.isEmpty() || frame.isEmpty()) return DepthStatistics(fraction, null, null, null, null, "UNKNOWN")
        val median = quantile(valid, .5f)
        val spread = quantile(frame, .99f) - quantile(frame, .01f)
        val hasVariation = spread > MIN_RELATIVE_SPREAD * maxOf(frame.last(), 1e-6f)
        val rank = if (hasVariation) (frame.count { it < median } + frame.count { it == median } * .5f) / frame.size else null
        val relation = when {
            fraction < MIN_VALID_FRACTION || rank == null -> "深度不确定"
            rank >= NEAR_PERCENTILE -> "相对较近"
            rank <= FAR_PERCENTILE -> "相对较远"
            else -> "场景中间范围"
        }
        return DepthStatistics(fraction, median, quantile(valid, .1f), quantile(valid, .9f), rank, relation)
    }
}
