package org.sightguard.app

import android.graphics.BitmapFactory
import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import kotlin.math.exp

/** Compare optimized traversal against the original full softmax on real model logits. */
class ArSemanticParityInstrumentedTest {
    @Test fun optimizedCellsMatchOriginalSoftmaxWithTwoThreads() {
        val ins=InstrumentationRegistry.getInstrumentation()
        val bitmap=ins.context.assets.open("validation.jpg").use {BitmapFactory.decodeStream(it)}!!
        try { SpatialDetector(ins.targetContext,2).use {detector ->
            val input=detector.prepare(bitmap);val logits=detector.infer(input.values)
            val expected=mutableListOf<SemanticCell>()
            for(pixel in 0 until 4096) {
                var best=0
                for(cls in 1 until 150) if(logits[cls][pixel]>logits[best][pixel]) best=cls
                val maximum=logits[best][pixel];var sum=0.0
                for(cls in 0 until 150) sum+=exp((logits[cls][pixel]-maximum).toDouble())
                val confidence=(1.0/sum).toFloat()
                if(!confidence.isFinite() || confidence<SpatialDetector.MIN_CONFIDENCE) continue
                val l=((pixel%64*4f-input.padLeft)/input.resizedWidth).coerceIn(0f,1f)
                val t=((pixel/64*4f-input.padTop)/input.resizedHeight).coerceIn(0f,1f)
                val r=(((pixel%64+1)*4f-input.padLeft)/input.resizedWidth).coerceIn(0f,1f)
                val b=(((pixel/64+1)*4f-input.padTop)/input.resizedHeight).coerceIn(0f,1f)
                if(r>l && b>t) expected.add(SemanticCell(l,t,r,b,best,confidence))
            }
            val actual=detector.analyzeAll(bitmap,SpatialFrameKey(1,0,0,0.0,bitmap.width,bitmap.height,SystemClock.elapsedRealtime())).cells
            assertTrue(expected.isNotEmpty());assertEquals(expected.size,actual.size)
            expected.zip(actual).forEach {(a,b) ->
                assertEquals(a.classId,b.classId);assertEquals(a.left,b.left,0f);assertEquals(a.top,b.top,0f)
                assertEquals(a.right,b.right,0f);assertEquals(a.bottom,b.bottom,0f)
                assertEquals(a.confidence,b.confidence,.000001f)
            }
        }} finally {bitmap.recycle()}
    }
}
