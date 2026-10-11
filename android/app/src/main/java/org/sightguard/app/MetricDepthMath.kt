package org.sightguard.app

import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.round

data class MetricShape(val height: Int, val width: Int)

/** Official lower-bound resize, rounded to multiples of 14. No letterbox or aspect warp. */
object MetricDepthMath {
    val supportedShapes = setOf(MetricShape(280,378), MetricShape(378,280),
        MetricShape(280,504), MetricShape(504,280))

    fun shape(width: Int, height: Int): MetricShape? {
        require(width > 0 && height > 0)
        val scale = max(280.0 / width, 280.0 / height)
        val resizedWidth = max(280, (round(width * scale / 14) * 14).toInt())
        val resizedHeight = max(280, (round(height * scale / 14) * 14).toInt())
        return MetricShape(resizedHeight,resizedWidth).takeIf { it in supportedShapes }
    }

    private fun cubic(value: Double): Double {
        val x = kotlin.math.abs(value)
        val a = -.75
        return when {
            x <= 1 -> ((a+2)*x-(a+3))*x*x+1
            x < 2 -> ((a*x-5*a)*x+8*a)*x-4*a
            else -> 0.0
        }
    }

    /** RGB double-valued cubic resampling before normalization, matching official OpenCV input. */
    fun prepare(pixels: IntArray, width: Int, height: Int, shape: MetricShape,
                output: FloatArray = FloatArray(3*shape.width*shape.height)): FloatArray {
        require(pixels.size == width*height && output.size == 3*shape.width*shape.height)
        val targetWidth=shape.width; val targetHeight=shape.height
        val xIndices=IntArray(targetWidth*4); val xWeights=DoubleArray(targetWidth*4)
        val yIndices=IntArray(targetHeight*4); val yWeights=DoubleArray(targetHeight*4)
        for(x in 0 until targetWidth) {
            val coordinate=(x+.5)*width/targetWidth-.5; val base=floor(coordinate).toInt()
            for(k in 0..3) {xIndices[x*4+k]=(base+k-1).coerceIn(0,width-1)
                xWeights[x*4+k]=cubic(coordinate-(base+k-1))}
        }
        for(y in 0 until targetHeight) {
            val coordinate=(y+.5)*height/targetHeight-.5; val base=floor(coordinate).toInt()
            for(k in 0..3) {yIndices[y*4+k]=(base+k-1).coerceIn(0,height-1)
                yWeights[y*4+k]=cubic(coordinate-(base+k-1))}
        }
        val plane=targetWidth*targetHeight
        for(y in 0 until targetHeight) for(x in 0 until targetWidth) {
            var red=0.0;var green=0.0;var blue=0.0
            for(j in 0..3) for(i in 0..3) {
                val value=pixels[yIndices[y*4+j]*width+xIndices[x*4+i]]
                val weight=xWeights[x*4+i]*yWeights[y*4+j]
                red+=((value ushr 16) and 255)*weight
                green+=((value ushr 8) and 255)*weight
                blue+=(value and 255)*weight
            }
            val index=y*targetWidth+x
            output[index]=((red/255-.485)/.229).toFloat()
            output[index+plane]=((green/255-.456)/.224).toFloat()
            output[index+plane*2]=((blue/255-.406)/.225).toFloat()
        }
        return output
    }

    fun pixelBounds(left: Float, right: Float, size: Int): IntRange =
        ceil(left*size-.5).toInt().coerceIn(0,size) until
            ceil(right*size-.5).toInt().coerceIn(0,size)
}
