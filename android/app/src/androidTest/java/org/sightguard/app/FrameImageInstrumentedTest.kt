package org.sightguard.app

import android.graphics.Color
import android.graphics.Rect
import androidx.camera.core.ImageInfo
import androidx.camera.core.ImageProxy
import java.lang.reflect.InvocationHandler
import java.lang.reflect.Proxy
import java.nio.ByteBuffer
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import androidx.test.ext.junit.runners.AndroidJUnit4

@RunWith(AndroidJUnit4::class)
class FrameImageInstrumentedTest {
    @Test
    fun cropRgbaStrideAndAllRightAngleRotationsPreservePixelCoordinates() {
        val crop = Rect(1, 1, 3, 4) // A 2x3 crop from a larger 4x5 plane.
        val expectedSourcesByRotation = mapOf(
            0 to listOf(1 to 1, 2 to 1, 1 to 2, 2 to 2, 1 to 3, 2 to 3),
            90 to listOf(1 to 3, 1 to 2, 1 to 1, 2 to 3, 2 to 2, 2 to 1),
            180 to listOf(2 to 3, 1 to 3, 2 to 2, 1 to 2, 2 to 1, 1 to 1),
            270 to listOf(2 to 1, 2 to 2, 2 to 3, 1 to 1, 1 to 2, 1 to 3),
        )

        for ((rotation, sourcePixels) in expectedSourcesByRotation) {
            val bitmap = uprightBitmap(fakeImageProxy(crop, rotation))
            val expectedWidth = if (rotation % 180 == 0) 2 else 3
            val expectedHeight = if (rotation % 180 == 0) 3 else 2
            assertEquals("width at $rotation degrees", expectedWidth, bitmap.width)
            assertEquals("height at $rotation degrees", expectedHeight, bitmap.height)

            sourcePixels.forEachIndexed { index, (sourceX, sourceY) ->
                val outputX = index % expectedWidth
                val outputY = index / expectedWidth
                assertEquals(
                    "pixel ($outputX,$outputY) at $rotation degrees",
                    sourceColor(sourceX, sourceY),
                    bitmap.getPixel(outputX, outputY),
                )
            }
            bitmap.recycle()
        }
    }

    private fun fakeImageProxy(crop: Rect, rotation: Int): ImageProxy {
        val width = 4
        val height = 5
        val pixelStride = 6 // RGBA plus two ignored padding bytes per pixel.
        val rowStride = width * pixelStride + 7
        val bufferOffset = 5
        val bytes = ByteBuffer.allocate(bufferOffset + height * rowStride + 8)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val base = bufferOffset + y * rowStride + x * pixelStride
                val color = sourceColor(x, y)
                bytes.put(base, Color.red(color).toByte())
                bytes.put(base + 1, Color.green(color).toByte())
                bytes.put(base + 2, Color.blue(color).toByte())
                bytes.put(base + 3, 0x7f.toByte())
                bytes.put(base + 4, 0x11.toByte())
                bytes.put(base + 5, 0x22.toByte())
            }
        }
        bytes.position(bufferOffset)

        val plane = proxy<ImageProxy.PlaneProxy> { method ->
            when (method.name) {
                "getBuffer" -> bytes
                "getRowStride" -> rowStride
                "getPixelStride" -> pixelStride
                else -> defaultValue(method.returnType)
            }
        }
        val imageInfo = proxy<ImageInfo> { method ->
            when (method.name) {
                "getRotationDegrees" -> rotation
                "getTimestamp" -> 1L
                else -> defaultValue(method.returnType)
            }
        }
        return proxy { method ->
            when (method.name) {
                "getWidth" -> width
                "getHeight" -> height
                "getPlanes" -> arrayOf(plane)
                "getCropRect" -> Rect(crop)
                "getImageInfo" -> imageInfo
                "getFormat" -> 0
                "getImage" -> null
                else -> defaultValue(method.returnType)
            }
        }
    }

    private fun sourceColor(x: Int, y: Int): Int = Color.rgb(
        20 + x * 30 + y * 7,
        40 + x * 13 + y * 19,
        60 + x * 11 + y * 17,
    )

    private inline fun <reified T> proxy(noinline value: (java.lang.reflect.Method) -> Any?): T =
        Proxy.newProxyInstance(
            T::class.java.classLoader,
            arrayOf(T::class.java),
            InvocationHandler { instance, method, args ->
                when (method.name) {
                    "toString" -> "FrameImageTestProxy(${T::class.java.simpleName})"
                    "hashCode" -> System.identityHashCode(instance)
                    "equals" -> instance === args?.firstOrNull()
                    else -> value(method)
                }
            },
        ) as T

    private fun defaultValue(type: Class<*>): Any? = when (type) {
        java.lang.Boolean.TYPE -> false
        java.lang.Byte.TYPE -> 0.toByte()
        java.lang.Short.TYPE -> 0.toShort()
        java.lang.Integer.TYPE -> 0
        java.lang.Long.TYPE -> 0L
        java.lang.Float.TYPE -> 0f
        java.lang.Double.TYPE -> 0.0
        java.lang.Character.TYPE -> '\u0000'
        else -> null
    }
}
