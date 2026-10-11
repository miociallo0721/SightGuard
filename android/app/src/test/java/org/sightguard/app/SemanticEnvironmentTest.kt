package org.sightguard.app

import org.junit.Assert.*
import org.junit.Test

class SemanticEnvironmentTest {
    private fun cells(id: Int=0, confidence: Float=.9f) = (0 until 20).flatMap { y -> (0 until 20).map { x ->
        SemanticCell(x/20f,y/20f,(x+1)/20f,(y+1)/20f,id,confidence)
    } }
    private fun evidence(time: Long, cells: List<SemanticCell> = cells(), generation: Long=1, epoch: Long=0) =
        GroundEvidence(SpatialFrameKey(generation,epoch,time.toInt(),time.toDouble(),640,480,time),cells,100,time+100,"semantics")
    private fun observe(engine: SemanticEnvironmentEngine,time: Long,cells: List<SemanticCell> = cells()) = engine.observe(evidence(time,cells),time+100,true,false)

    @Test fun confirmsOnlyAcrossDistinctValidSourceObservationsAndNeverRepeatsAfterPlayback() {
        val e=SemanticEnvironmentEngine()
        assertTrue(observe(e,1000).isEmpty())
        assertTrue(observe(e,1000).isEmpty())
        val notice=observe(e,1700).single()
        assertEquals(EnvironmentKind.WALL,notice.kind)
        e.playbackStarted(notice.id)
        for(t in 2400L..10000L step 700) assertTrue(observe(e,t).isEmpty())
        assertTrue(e.status.contains("已确认")); assertFalse(e.approachEnabled)
    }
    @Test fun absentSmallPeripheralLowConfidenceAndInvalidMasksDoNotCreateEvents() {
        for(mask in listOf(emptyList(),cells(14),cells(4),cells(0,.4f),cells().filter{it.right<=.25f},
            listOf(SemanticCell(Float.NaN,0f,1f,1f,0,.9f)))) {
            val e=SemanticEnvironmentEngine()
            assertTrue(observe(e,1000,mask).isEmpty()); assertTrue(observe(e,1700,mask).isEmpty())
        }
    }
    @Test fun stairsStepAndPoleHaveSeparateEventsAndDoorTreeRemainOverviewOnly() {
        for((id,kind) in listOf(53 to EnvironmentKind.STAIRS,59 to EnvironmentKind.STAIRS,121 to EnvironmentKind.STEP,93 to EnvironmentKind.POLE)) {
            val e=SemanticEnvironmentEngine(); observe(e,1000,cells(id))
            val second=observe(e,1700,cells(id))
            if(id==121) { assertTrue(second.isEmpty()); assertEquals(kind,observe(e,2400,cells(id)).single().kind) }
            else assertEquals(kind,second.single().kind)
        }
    }
    @Test fun staleFutureRapidAndSceneMismatchInvalidatePendingWithoutInventingEvidence() {
        for(mode in 0..3) {
            val e=SemanticEnvironmentEngine(); observe(e,1000); observe(e,1700)
            val ev=evidence(2400)
            assertTrue(e.observe(ev,when(mode){0->3000;1->2300;else->2500},mode!=2,mode==3).isEmpty())
            assertTrue(e.pending(2500).isEmpty())
            assertTrue(observe(e,3100).isEmpty())
            assertEquals(1,observe(e,3800).size)
        }
    }
    @Test fun recoveryRetainsIdentityThenExplicitEndAndCooldownAllowANewEvent() {
        val e=SemanticEnvironmentEngine(); observe(e,1000)
        val first=observe(e,1700).single(); e.playbackStarted(first.id)
        observe(e,2400,emptyList()); assertTrue(observe(e,3100).isEmpty())
        assertTrue(e.validIds(3200).contains(first.id))
        observe(e,8000,emptyList()); assertTrue(e.validIds(8100).isEmpty())
        observe(e,8700); assertTrue(observe(e,9400).isEmpty())
        observe(e,14000,emptyList()); observe(e,18000)
        assertNotEquals(first.id,observe(e,18700).single().id)
    }
    @Test fun cancelledOrMutedUnplayedEventsRetryOnlyWithFreshEvidence() {
        val e=SemanticEnvironmentEngine(); observe(e,1000)
        val first=observe(e,1700).single()
        assertTrue(e.pending(2200).isEmpty())
        assertEquals(first.id,observe(e,2400).single().id)
        e.playbackStarted(first.id); assertTrue(observe(e,3100).isEmpty())
    }
    @Test fun stopSeekSessionAndEmptyObservationAreIsolated() {
        val e=SemanticEnvironmentEngine(); observe(e,1000); val n=observe(e,1700).single()
        e.playbackStarted(n.id)
        assertTrue(e.observe(evidence(100,epoch=1),200,true,false).isEmpty())
        assertEquals(1,e.observe(evidence(800,epoch=1),900,true,false).size)
        e.reset(); assertTrue(e.pending(1000).isEmpty())
        assertTrue(observe(e,1700,emptyList()).isEmpty())
    }
    @Test fun sourceTimeNotInferenceSpeedControlsConfirmationAndExpiry() {
        val e=SemanticEnvironmentEngine(); observe(e,1000)
        assertTrue(observe(e,1100).isEmpty())
        assertEquals(1,observe(e,1400).size)
        assertTrue(e.pending(1900).isEmpty()); e.pending(6000)
        assertEquals("环境：事件已结束",e.status)
    }
    @Test fun imageContinuityRejectsChangedAndMalformedFrames() {
        assertTrue(SemanticSceneContinuity.matches(floatArrayOf(.5f,.5f),floatArrayOf(.55f,.5f),.12f))
        assertFalse(SemanticSceneContinuity.matches(floatArrayOf(0f,0f),floatArrayOf(1f,1f),.12f))
        assertFalse(SemanticSceneContinuity.matches(floatArrayOf(Float.NaN),floatArrayOf(0f),.12f))
        assertFalse(SemanticSceneContinuity.matches(floatArrayOf(),floatArrayOf(),.12f))
    }
    @Test fun exactChineseFallbackAndYoloPriorityArePreserved() {
        for(kind in EnvironmentKind.entries) assertFalse(SpeechPromptCatalog.resolve(kind.text).usedGeneric)
        val wall=SpeechRequest(EnvironmentKind.WALL.text,false,"semantic:wall",500)
        assertTrue(SpeechRequestRules.preempts(SpeechRequest("前方有行人",false,"risk",500),wall))
        assertFalse(SpeechRequestRules.preempts(wall,SpeechRequest("前方有行人",true,"risk",500)))
    }
}
