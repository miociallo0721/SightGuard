package org.sightguard.app

data class ImageFit(val scale: Float, val offsetX: Float, val offsetY: Float) {
    fun x(pixelX: Float) = offsetX + pixelX * scale
    fun y(pixelY: Float) = offsetY + pixelY * scale
}
/** Shared FIT_CENTER mapping for upright camera crop and the exact decoded video bitmap. */
fun fitImage(sourceWidth: Int, sourceHeight: Int, viewWidth: Float, viewHeight: Float): ImageFit {
    require(sourceWidth > 0 && sourceHeight > 0 && viewWidth > 0 && viewHeight > 0)
    val scale = minOf(viewWidth / sourceWidth, viewHeight / sourceHeight)
    return ImageFit(scale, (viewWidth - sourceWidth * scale) / 2, (viewHeight - sourceHeight * scale) / 2)
}
