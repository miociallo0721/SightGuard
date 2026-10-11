package org.sightguard.app

import org.junit.Assert.*
import org.junit.Test

class SpatialGridTest {
    private val key=SpatialFrameKey(1,0,7,100.0,811,1081,1000)
    private val stable=MotionRules.classify(.02f,1L,true)
    @Test fun missingAndMismatchedSourcesRemainUnknown() {
        val empty=SpatialGridFusion().fuse(key,null,null,null,MotionEvidence(MotionQuality.UNKNOWN,"test"))
        assertEquals(225,empty.cells.size); assertTrue(empty.cells.all { it.state=="UNKNOWN" }); assertEquals("PARTIAL",empty.state)
        val old=GridSemanticSource(key.copy(frameIndex=6),listOf(SemanticCell(0f,0f,1f,1f,3,.9f)),"old")
        val result=SpatialGridFusion().fuse(key,old,null,null,stable)
        assertTrue(result.cells.all { it.ground==null }); assertTrue(result.failures.any { "mismatch" in it })
    }
    @Test fun sameFrameGroundOccupancyConflictAndProvenance() {
        val semantic=GridSemanticSource(key,listOf(SemanticCell(0f,0f,1f,1f,6,.9f)),"semantic-sha")
        val objects=GridDetectionSource(key,listOf(Detection(0,.9f,.4f,.4f,.6f,.7f)),"yolo-sha")
        val depth=GridDepthSource(key,15,15,FloatArray(225) { it.toFloat() },"depth-sha")
        val result=SpatialGridFusion().fuse(key,semantic,depth,objects,stable)
        assertTrue(result.cells.any { it.state=="CONFLICT" }); assertTrue(result.cells.any { it.state=="GROUND" })
        assertTrue(result.cells.all { it.sources.keys==setOf("semantic","detection","depth") })
        assertFalse(result.productionDecisionChanged); assertEquals("SAME_FRAME_RESEARCH",result.state)
        assertEquals(1f,result.cells.last().right,0f); assertEquals(1f,result.cells.last().bottom,0f)
        assertTrue(result.cells.any { it.depth?.relation=="相对较近" })
    }
    @Test fun canopyNeverBecomesOccupancyAndRapidDegrades() {
        val semantic=GridSemanticSource(key,listOf(SemanticCell(0f,0f,1f,.4f,4,.9f)),"tree")
        val rapid=MotionRules.classify(2f,1L,true)
        val result=SpatialGridFusion().fuse(key,semantic,null,GridDetectionSource(key,emptyList(),"yolo"),rapid)
        assertTrue(result.cells.none { it.state=="OBJECT" || it.state=="CONFLICT" }); assertTrue(result.cells.all { it.quality=="DEGRADED" })
    }
    @Test fun motionClockGateStalenessTiltAndThresholds() {
        assertEquals(MotionQuality.UNKNOWN,MotionRules.classify(4f,1L,false).quality)
        assertEquals(MotionQuality.LIMITED,MotionRules.classify(.1f,400_000_000L,true).quality)
        assertEquals(MotionQuality.STABLE,stable.quality)
        assertEquals(MotionQuality.MODERATE,MotionRules.classify(.3f,1L,true).quality)
        assertEquals(MotionQuality.RAPID,MotionRules.classify(2f,1L,true).quality)
        assertEquals(MotionQuality.STABLE,MotionRules.classify(.1f,1L,true,80f).quality) // Upright camera pitch is not an abnormal absolute pose.
        assertEquals(MotionQuality.RAPID,MotionRules.classify(.1f,1L,true,orientationDeltaDegrees=40f).quality)
    }
    @Test fun staleDepthDimensionsAndInvalidPixelsCannotInventNearRelation() {
        val malformed=GridDepthSource(key,15,15,FloatArray(1),"wrong")
        val bad=SpatialGridFusion().fuse(key,null,malformed,null,stable)
        assertTrue(bad.cells.all { it.depth==null }); assertTrue("depth:invalid_dimensions" in bad.failures)
        val constant=GridDepthSource(key,15,15,FloatArray(225){1f},"constant")
        val output=SpatialGridFusion().fuse(key,null,constant,null,stable)
        assertTrue(output.cells.all { it.depth?.framePercentile==null })
    }
}
