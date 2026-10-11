package org.sightguard.app

import org.junit.Assert.*
import org.junit.Test

class SpatialCoordinatesTest {
    @Test fun portraitLandscapeAndOddDimensionsRoundTrip() {
        for ((w,h) in listOf(480 to 640, 640 to 480, 811 to 1081, 1 to 30)) for (size in listOf(256,280)) {
            val t = SquareTransform(w,h,size)
            for (x in listOf(0f,.2f,.5f,1f)) { assertEquals(x,t.sourceX(t.modelX(x)),.00001f); assertEquals(x,t.sourceY(t.modelY(x)),.00001f) }
            assertTrue(t.resizedWidth <= size && t.resizedHeight <= size)
        }
    }
    @Test fun robustRelativeComparisonAndInvalidValues() {
        val scene = FloatArray(100) { it.toFloat() }
        assertEquals("相对较近",RelativeDepthRules.summarize(floatArrayOf(80f,85f,90f),scene).relation)
        assertEquals("相对较远",RelativeDepthRules.summarize(floatArrayOf(1f,4f,8f),scene).relation)
        assertEquals("深度不确定",RelativeDepthRules.summarize(floatArrayOf(2f,Float.NaN),scene).relation)
        assertNull(RelativeDepthRules.summarize(FloatArray(4){2f},FloatArray(10){2f}).framePercentile)
        assertEquals("UNKNOWN",RelativeDepthRules.summarize(floatArrayOf(Float.POSITIVE_INFINITY),scene).relation)
        assertEquals(85f,RelativeDepthRules.summarize(floatArrayOf(80f,85f,90f),scene).median!!,0f)
    }
}
