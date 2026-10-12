package org.sightguard.app

import org.junit.Assert.*
import org.junit.Test

class LocalSpaceCandidateTest {
    private fun sample(i: Int, cols: IntRange = 0..4, generation: Long=1, detections: List<Detection> = emptyList(),
        quality: MotionQuality=MotionQuality.UNKNOWN, gapRow: Int?=null): Pair<SpatialGridEvidence,GroundEvidence> {
        val key=SpatialFrameKey(generation,0,i,i*800.0,300,300,i*800L)
        val cells=(0..14).flatMap {r-> (0..14).map {c-> SemanticCell(c/15f,r/15f,(c+1)/15f,(r+1)/15f,if(c in cols && r>=8 && r!=gapRow) 3 else 0,.95f)}}
        val s=GroundEvidence(key,cells,20,key.capturedAtMs+20,"model")
        return SpatialGridFusion(GridConfig(minConfidence=.65f,includeFurniture=true)).fuse(key,GridSemanticSource(key,cells,s.modelSha),null,GridDetectionSource(key,detections,"actual"),MotionEvidence(quality,"test")) to s
    }
    private fun observe(e: LocalSpaceCandidateEvaluator,i: Int,cols: IntRange=0..4): LocalSpaceRecord {
        val (g,s)=sample(i,cols);return e.observe(g,s,g.key.capturedAtMs+30,true)
    }
    @Test fun confirmsOnceAndExpiresWithoutRepeating() {
        val e=LocalSpaceCandidateEvaluator(); assertEquals("CONFIRMING",observe(e,1).status); observe(e,2)
        val r=observe(e,3);assertEquals("SUPPORTED_CANDIDATE",r.status);assertEquals(Direction.LEFT,r.direction)
        assertNotNull(e.pending(2430));e.playbackStarted(r.eventId!!,2430);assertNull(e.pending(2450))
        assertEquals(r.eventId,observe(e,4).eventId);assertNull(e.pending(3240));assertNull(e.pending(5000))
    }
    @Test fun tiesAndDisconnectedGroundNeverSuggestDirection() {
        assertEquals("TIED",observe(LocalSpaceCandidateEvaluator(),1,0..14).status)
        val (g,s)=sample(1,gapRow=11);assertEquals("INSUFFICIENT_EVIDENCE",LocalSpaceCandidateEvaluator().observe(g,s,830,true).status)
    }
    @Test fun staleMismatchRapidUnknownAndSceneChange() {
        val (g,s)=sample(1);val e=LocalSpaceCandidateEvaluator()
        assertEquals("ABSTAIN",e.observe(g,s,2000,true).status)
        assertEquals("ABSTAIN",e.observe(g,s.copy(key=s.key.copy(epoch=9)),830,true).status)
        assertEquals("ABSTAIN",e.observe(g,s,830,false).status)
        assertEquals("ABSTAIN",e.observe(g.copy(motion=MotionEvidence(MotionQuality.RAPID,"turn")),s,830,true).status)
        assertEquals("ABSTAIN",e.observe(g,s.copy(modelSha="fake"),830,true).status)
        assertEquals("INSUFFICIENT_EVIDENCE",LocalSpaceCandidateEvaluator().observe(g.copy(cells=g.cells.map {it.copy(state="UNKNOWN",ground=null)}),s,830,true).status)
    }
    @Test fun duplicateSessionGapAndDirectionChangeResetConfirmation() {
        val e=LocalSpaceCandidateEvaluator();observe(e,1);observe(e,2)
        assertEquals("ABSTAIN",observe(e,2).status);assertEquals(1,observe(e,3).support)
        assertEquals(1,observe(e,10).support);assertEquals(1,observe(e,11,10..14).support)
        val (g,s)=sample(12,generation=2);assertEquals(1,e.observe(g,s,9630,true).support)
    }
    @Test fun obstacleAndStairEvidenceVeto() {
        val (g,s)=sample(1,detections=listOf(Detection(56,.9f,0f,.55f,.34f,1f)))
        assertEquals("INSUFFICIENT_EVIDENCE",LocalSpaceCandidateEvaluator().observe(g,s,830,true).status)
        val (clear,sem)=sample(1)
        assertEquals("INSUFFICIENT_EVIDENCE",LocalSpaceCandidateEvaluator().observe(clear,sem.copy(cells=sem.cells+SemanticCell(0f,.7f,.34f,.8f,121,.7f)),830,true).status)
        val engine=LocalSpaceCandidateEvaluator();observe(engine,1);observe(engine,2);observe(engine,3)
        assertTrue(engine.obstructed(listOf(Detection(0,.8f,.1f,.7f,.3f,.9f))))
        assertFalse(engine.obstructed(listOf(Detection(0,.8f,.6f,.7f,.9f,.9f))))
    }
    @Test fun cooldownAndSpeechPriority() {
        val e=LocalSpaceCandidateEvaluator();observe(e,1);observe(e,2);val r=observe(e,3);e.playbackStarted(r.eventId!!,2430)
        e.invalidate();observe(e,4,10..14);observe(e,5,10..14);observe(e,6,10..14);assertNull(e.pending(4830))
        val local=SpeechRequest("candidate",false,"local-space:1",5000)
        assertTrue(SpeechRequestRules.preempts(SpeechRequest("wall",false,"semantic:1",5000),local))
        assertTrue(SpeechRequestRules.preempts(SpeechRequest("risk",true,"risk",5000),local))
    }
    @Test fun sparseGridMatchesBruteOverlapOnOddImageDimensions() {
        val random=java.util.Random(42)
        val key=SpatialFrameKey(1,0,1,100.0,641,479,100)
        val cells=(0..299).map {
            val l=random.nextFloat();val t=random.nextFloat()
            SemanticCell(l,t,(l+random.nextFloat()*.3f).coerceAtMost(1f),(t+random.nextFloat()*.3f).coerceAtMost(1f),if(it%2==0) 3 else 0,random.nextFloat())
        }
        val grid=SpatialGridFusion(GridConfig(minConfidence=.65f)).fuse(key,GridSemanticSource(key,cells,"test"),null,GridDetectionSource(key,emptyList(),"test"),MotionEvidence(MotionQuality.UNKNOWN,"test"))
        for(cell in grid.cells) for(cls in listOf(0,3)) {
            val area=(cell.right-cell.left)*(cell.bottom-cell.top)
            val expected=cells.filter {it.classId==cls && it.confidence>=.65f}.sumOf {
                (maxOf(0f,minOf(cell.right,it.right)-maxOf(cell.left,it.left))*maxOf(0f,minOf(cell.bottom,it.bottom)-maxOf(cell.top,it.top))/area).toDouble()
            }.coerceAtMost(1.0)
            assertEquals(expected,(if(cls==0) cell.wall!! else cell.ground!!).toDouble(),.00001)
        }
    }
}
