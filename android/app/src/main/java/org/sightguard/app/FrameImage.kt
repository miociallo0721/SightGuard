package org.sightguard.app

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import androidx.camera.core.ImageProxy

/** RGBA plane respects row and pixel stride; rotate before model and overlay coordinates. */
internal fun uprightBitmap(image: ImageProxy): Bitmap {
    val plane = image.planes[0]
    val buffer = plane.buffer.duplicate()
    val offset = buffer.position()
    val crop = image.cropRect
    val pixels = IntArray(crop.width() * crop.height())
    for (y in 0 until crop.height()) for (x in 0 until crop.width()) {
        val base = offset + (y + crop.top) * plane.rowStride + (x + crop.left) * plane.pixelStride
        // CameraX RGBA_8888 memory order: R, G, B, A.
        pixels[y * crop.width() + x] = Color.rgb(buffer.get(base).toInt() and 255, buffer.get(base + 1).toInt() and 255, buffer.get(base + 2).toInt() and 255)
    }
    val raw = Bitmap.createBitmap(pixels, crop.width(), crop.height(), Bitmap.Config.ARGB_8888)
    if (image.imageInfo.rotationDegrees == 0) return raw
    val upright = Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, Matrix().apply { postRotate(image.imageInfo.rotationDegrees.toFloat()) }, false)
    if (upright !== raw) raw.recycle()
    return upright
}
