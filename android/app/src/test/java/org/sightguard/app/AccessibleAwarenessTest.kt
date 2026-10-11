package org.sightguard.app

import org.junit.Assert.*
import org.junit.Test

class AccessibleAwarenessTest {
    @Test fun pixelTrackingRecoversKnownSyntheticWarpWithoutPhysicalMotionClaim() {
        val w=128;val h=96
        // Checkerboard intersections provide actual Shi–Tomasi corners, unlike smooth stripes.
        val original=GrayImage(w,h,FloatArray(w*h) {i->if(((i%w)/6+(i/w)/6)%2==0) .1f else .9f})
        val warped=GrayImage(w,h,FloatArray(w*h) {i->original.at((i%w)-1f,(i/w)-1f)})
        val result=WallFeatureFlow(WallResearchConfig(maxFlowMs=100)).analyze(original,warped,mask(0))
        assertFalse(result.budgetExceeded);assertTrue("Real pixel LK tracks: ${result.inliers}",result.inliers>=16)
        assertEquals(1f,result.scale!!,.01f);assertEquals(0f,result.rotationRad!!,.01f)
        assertTrue(result.residual!!<.5f)
    }
    @Test fun weakStairsCannotPromoteStrongStepToValidatedStairs() {
        val cells=mask(121).map {if(it.left<.5f) it.copy(classId=59,confidence=.4f) else it}
        val parts=SemanticStructureMeasure(EnvironmentConfig()).components(cells)
        assertTrue(parts.none {it.kind==EnvironmentKind.STAIRS})
        assertEquals(setOf(121),parts.single {it.supported}.rawClassIds)
    }
    @Test fun sparseSemanticGridMatchesFullGridWithOddPixelPartitions() {
        val k=SpatialFrameKey(1,0,1,1000.0,641,479,1000)
        val cells=mask(0).map {if(it.top>.5f) it.copy(classId=3) else it}
        val fusion=SpatialGridFusion();val summary=fusion.semanticSummary(k,cells)
        val full=fusion.fuse(k,GridSemanticSource(k,cells,"synthetic equivalence"),null,null,MotionEvidence(MotionQuality.UNKNOWN,"test"))
        assertEquals(full.cells.count {it.state=="WALL"},summary.wallCells)
        assertEquals(full.cells.count {it.state=="GROUND"},summary.groundCells)
        assertTrue(summary.boundaryAdjacencies>0)
    }
    @Test fun unvalidatedStepCannotBecomeAutomaticAndAbsentObservationResetsConfirmation() {
        val e=SemanticEnvironmentEngine(EnvironmentConfig(automaticKinds=EnvironmentKind.entries.toSet()-EnvironmentKind.STEP))
        for(t in listOf(1000L,1700L,2400L,3100L)) assertTrue(observe(e,t,mask(121)).isEmpty())
        assertTrue(e.supports.any {it.kind==EnvironmentKind.STEP && it.supported})
        assertTrue(e.silenceReason.contains("尚未验证"))
        val wall=SemanticEnvironmentEngine();observe(wall,1000,mask(0));observe(wall,1700,emptyList())
        assertTrue(observe(wall,2400,mask(0)).isEmpty())
        assertEquals(EnvironmentKind.WALL,observe(wall,3100,mask(0)).single().kind)
    }
    private fun mask(id: Int)= (0 until 40).flatMap { y->(0 until 40).map {x->SemanticCell(x/40f,y/40f,(x+1)/40f,(y+1)/40f,id,.9f)} }
    private fun observe(e: SemanticEnvironmentEngine,t: Long,c: List<SemanticCell>) =
        e.observe(GroundEvidence(SpatialFrameKey(1,0,t.toInt(),t.toDouble(),640,480,t),c,50,t+50,"synthetic state test"),t+50,true,false)
    @Test fun disconnectedSpecklesCannotBorrowAggregateArea() {
        val cells=mask(53).filter { ((it.left*40).toInt()+(it.top*40).toInt())%2==0 }
        val e=SemanticEnvironmentEngine();observe(e,1000,cells)
        assertTrue(observe(e,1700,cells).isEmpty());assertTrue(e.supports.none {it.supported})
    }
    @Test fun adjacentStairLabelsAggregateOnceWithRawSources() {
        val cells=mask(53).map {if(it.left>.5f) it.copy(classId=121) else if(it.top>.5f) it.copy(classId=59) else it}
        val e=SemanticEnvironmentEngine();observe(e,1000,cells)
        val notice=observe(e,1700,cells).single()
        assertEquals(EnvironmentKind.STAIRS,notice.kind);assertEquals(setOf(53,59,121),notice.rawClassIds)
        e.playbackStarted(notice.id);assertTrue(observe(e,2400,mask(121)).isEmpty())
    }
    @Test fun twoSeparatedColumnsHaveIndependentEventsAndDirections() {
        val cells=mask(42).filter {it.left in .125f.. .25f || it.left in .7f.. .825f}
        val e=SemanticEnvironmentEngine();observe(e,1000,cells);val notices=observe(e,1700,cells)
        assertEquals(2,notices.size);assertEquals(2,notices.map {it.id}.toSet().size)
        assertEquals(setOf(Direction.LEFT,Direction.RIGHT),notices.map {it.direction}.toSet())
        assertTrue(notices.all {it.rawClassIds==setOf(42)})
    }
    @Test fun stepRequiresThreeStrongObservationsWhileOverviewDoesNotAlert() {
        val e=SemanticEnvironmentEngine();observe(e,1000,mask(121));assertTrue(observe(e,1700,mask(121)).isEmpty())
        assertEquals(EnvironmentKind.STEP,observe(e,2400,mask(121)).single().kind)
        for(id in listOf(14,32,4,87,96,21,34)) {val n=SemanticEnvironmentEngine();observe(n,1000,mask(id));assertTrue(observe(n,1700,mask(id)).isEmpty())}
    }
    @Test fun exactDirectionalAudioDoesNotFallBackToGenericTarget() {
        for(kind in EnvironmentKind.entries) for(direction in Direction.entries) {
            val text=environmentText(kind,direction);val selection=SpeechPromptCatalog.resolve(text)
            assertFalse(text,selection.usedGeneric);assertEquals(text,selection.spokenText)
        }
    }
    @Test fun schedulerOrdersRiskStructureDescriptionAndStatus() {
        fun r(id: String,important: Boolean=false)=SpeechRequest("text",important,id,1000)
        val risk=r("risk",true);val structure=r("semantic:stairs");val overview=r("overview-1");val status=r("status-1")
        assertTrue(SpeechRequestRules.preempts(risk,structure));assertTrue(SpeechRequestRules.preempts(structure,overview))
        assertTrue(SpeechRequestRules.preempts(overview,status));assertFalse(SpeechRequestRules.preempts(overview,structure))
    }
    private fun key(t: Long,epoch: Long=0)=SpatialFrameKey(1,epoch,t.toInt(),t.toDouble(),192,144,t,t*1_000_000,"REALTIME")
    private fun motion(t: Long)=MotionEvidence(MotionQuality.STABLE,"synthetic aligned IMU",0f,synchronized=true,sampleCount=40,intervalStartNs=(t-700)*1_000_000,intervalEndNs=(t+50)*1_000_000,rotationMagnitudeRad=0f)
    private fun fit(scale: Float=1f,translation: Float=0f): FlowFit {
        val points=(0..3).flatMap {y->(0..4).map {x->val px=25f+x*30;val py=20f+y*30;FlowPoint(px,py,96+(px-96)*scale+translation,72+(py-72)*scale,0f)}}
        return WallFeatureFlow().fit(points,192,144)
    }
    @Test fun staticWallIsPresenceWithoutNearOrApproach() {
        val rules=WallApproachRules();rules.evaluate(key(1000),1050,true,null,motion(1000),"unavailable")
        val r=rules.evaluate(key(1700),1750,true,fit(),motion(1700),"same_frame_relative_only")
        assertEquals(WallEvidenceState.WALL_PRESENT,r.state);assertTrue(r.wallPresent==true)
        assertFalse(r.possibleNear);assertFalse(r.approachingCandidate);assertFalse(r.automaticAlertEnabled)
    }
    @Test fun consistentExpansionIsShadowOnlyAndNeverBodyMotionClaim() {
        val rules=WallApproachRules();rules.evaluate(key(1000),1050,true,null,motion(1000),"unavailable")
        var r=rules.evaluate(key(1700),1750,true,fit(1.03f),motion(1700),"relative")
        for(t in listOf(2400L,3100L)) r=rules.evaluate(key(t),t+50,true,fit(1.03f),motion(t),"relative")
        assertTrue(r.approachingCandidate);assertEquals(3,r.temporalSupport)
        assertEquals(WallEvidenceState.ABSTAIN,r.state);assertEquals("camera_translation_unverified_body_motion_unknown",r.abstainReason)
        assertFalse(r.possibleNear);assertFalse(r.automaticAlertEnabled)
    }
    @Test fun rotationLateralMissingTextureStaleAndEpochChangesAbstain() {
        for(mode in 0..5) {
            val rules=WallApproachRules();rules.evaluate(key(1000),1050,true,null,motion(1000),"unavailable")
            val r=rules.evaluate(key(1700,if(mode==5) 1 else 0),if(mode==3) 2300 else 1750,true,
                if(mode==2) fit().copy(inliers=3) else if(mode==1) fit(1.03f,20f) else fit(1.03f),
                if(mode==0) motion(1700).copy(quality=MotionQuality.RAPID,gyroRadPerSecond=2f) else if(mode==4) motion(1700).copy(intervalStartNs=null) else motion(1700),"unavailable")
            assertFalse(r.approachingCandidate);assertNotNull(r.abstainReason);assertFalse(r.possibleNear)
        }
    }
    @Test fun lowTextureOpticalFlowHasNoInventedFeatures() {
        val flat=GrayImage(96,72,FloatArray(96*72){.5f})
        val result=WallFeatureFlow().analyze(flat,flat,mask(0))
        assertEquals(0,result.inliers);assertNull(result.scale)
    }
    @Test fun ransacRejectsRandomOutliersAndRecoversKnownImageScale() {
        val good=fit(1.03f);assertEquals(1.03f,good.scale!!,.001f);assertEquals(4,good.coverage)
        val outliers=(0..5).map {i->FlowPoint(10f+i*25,10f,170f-i*20,130f,0f)}
        val result=WallFeatureFlow().fit(good.tracks+outliers,192,144)
        assertEquals(1.03f,result.scale!!,.001f);assertEquals(good.tracks.size,result.inliers)
    }
}
