package org.sightguard.app

import org.junit.Assert.*
import org.junit.Test

class ArSpatialEvidenceTest {
    private fun sample(i: Int,z: Float=0f,tracking: String="TRACKING",depthDelayMs: Long=0,forward: Vec3=Vec3(0f,0f,-1f),cameraX: Float=0f): Pair<GroundEvidence,ArGeometry> {
        val time=1000L+i*600;val stamp=time*1000000
        val key=SpatialFrameKey(1,0,i,time.toDouble(),480,640,time,stamp,"ARCORE",90)
        val points=(0..8).flatMap {y -> (0..8).map {x ->
            ArDepthPoint(.28f+x*.05f,.25f+y*.075f,1.2f+z,.95f,Vec3((x-4)*.1f,(y-4)*.1f,-1.2f))
        } }
        return GroundEvidence(key,listOf(SemanticCell(0f,0f,1f,1f,0,.95f)),100,time+100,"VALID") to
            ArGeometry(stamp,stamp-depthDelayMs*1000000,time,90,tracking,"NONE",Vec3(cameraX,0f,z),forward,points)
    }
    private fun eval(engine: ArSpatialEvaluator,pair: Pair<GroundEvidence,ArGeometry>,age: Long=150,scene: Boolean=true)=
        engine.evaluate(pair.first.key,pair.first,pair.second,emptyList(),pair.first.key.capturedAtMs+age,scene)
    @Test fun realGeometryContractConfirmsNearOnlyAfterThreeObservations() {
        val e=ArSpatialEvaluator();assertFalse(eval(e,sample(0)).near);assertFalse(eval(e,sample(1)).near)
        val r=eval(e,sample(2));assertTrue(r.near);assertEquals(1.2f,r.wallDistanceM!!,.001f);assertFalse(r.approaching)
    }
    @Test fun cameraApproachNeedsTranslationAndConsistentPlaneDistance() {
        val e=ArSpatialEvaluator();eval(e,sample(0));eval(e,sample(1,-.1f));assertTrue(eval(e,sample(2,-.25f)).approaching)
    }
    @Test fun rotationLateralAndRecedingAreNotApproach() {
        for(kind in 0..2) {
            val e=ArSpatialEvaluator();eval(e,sample(0));eval(e,sample(1))
            val r=when(kind) {0 -> eval(e,sample(2,-.25f,forward=Vec3(.4f,0f,-.9165f)));1 -> eval(e,sample(2,cameraX=.3f));else -> eval(e,sample(2,.3f))}
            assertFalse(r.approaching)
        }
    }
    @Test fun staleLostTrackingSceneChangeAndMissingDepthAbstain() {
        for(pair in listOf(sample(0,tracking="PAUSED"),sample(0,depthDelayMs=300))) assertNull(eval(ArSpatialEvaluator(),pair).wallDistanceM)
        assertNull(eval(ArSpatialEvaluator(),sample(0),age=501).wallDistanceM)
        assertNull(eval(ArSpatialEvaluator(),sample(0),scene=false).wallDistanceM)
    }
    @Test fun mismatchedFramesRejectedAndDuplicateObservationsDoNotConfirm() {
        val e=ArSpatialEvaluator();val pair=sample(0)
        repeat(4) {assertFalse(eval(e,pair).near)}
        assertNull(eval(e,pair.first to pair.second.copy(timestampNs=1)).wallDistanceM)
    }
    @Test fun blankLowConfidenceOrMixedPlaneCannotBecomeWallDistance() {
        val pair=sample(0)
        for(points in listOf(emptyList(),pair.second.points.map {it.copy(confidence=.2f)},pair.second.points.mapIndexed {i,p ->p.copy(world=p.world.copy(z=p.world.z+i*.02f*(i%3)))}))
            assertNull(eval(ArSpatialEvaluator(),pair.first to pair.second.copy(points=points)).wallDistanceM)
    }
    @Test fun invalidObservationDoesNotCountAsTemporalSupport() {
        val e=ArSpatialEvaluator();eval(e,sample(0,tracking="PAUSED"));eval(e,sample(1,depthDelayMs=300));assertFalse(eval(e,sample(2)).near)
    }
    @Test fun directionMappingsRoundTripAllImageRotations() {
        for(rotation in listOf(0,90,180,270)) for(p in listOf(.1f to .8f,.9f to .2f)) {
            val rotated=ArGeometryRules.rotate(p.first,p.second,rotation);val restored=ArGeometryRules.unrotate(rotated.first,rotated.second,rotation)
            assertEquals(p.first,restored.first,.0001f);assertEquals(p.second,restored.second,.0001f)
        }
    }
    @Test fun playbackAcknowledgementDeduplicatesAndFailedPlaybackCanRetry() {
        val e=ArSpatialEvaluator();eval(e,sample(0));eval(e,sample(1));val r=eval(e,sample(2));val now=r.generatedAtMs
        val policy=ArSpeechPolicy();val n=policy.notices(r,now).single();policy.offered(n.id)
        assertTrue(policy.notices(r,now).isEmpty());policy.playback(n.id,false,now);assertEquals(1,policy.notices(r,now).size)
        policy.playback(n.id,true,now);assertTrue(policy.notices(r,now).isEmpty());assertTrue(policy.notices(r,now+501).isEmpty())
    }
    @Test fun researchCannotPreemptFormalSemanticOrRiskSpeech() {
        val research=SpeechRequest("near",false,"research:near",1000)
        assertFalse(SpeechRequestRules.preempts(research,SpeechRequest("wall",false,"semantic:wall",1000)))
        assertTrue(SpeechRequestRules.preempts(SpeechRequest("risk",true,"risk",1000),research))
    }
    @Test fun exactChineseResearchResourcesNeverUseGenericFallback() {
        for(text in listOf("前方墙面可能较近，请注意。","相机可能正在靠近前方墙面，请注意。","画面左侧可见地面较连续，通行条件尚未确认。","画面中央可见地面较连续，通行条件尚未确认。","画面右侧可见地面较连续，通行条件尚未确认。")) {
            val prompt=SpeechPromptCatalog.resolve(text);assertFalse(prompt.usedGeneric);assertEquals(text,prompt.spokenText)
        }
    }
    private fun groundSample(i: Int,bothSides: Boolean=false): Pair<GroundEvidence,ArGeometry> {
        val pair=sample(i);val ranges=if(bothSides) listOf(.05f to .35f,.65f to .95f) else listOf(.05f to .35f)
        val cells=ranges.map {SemanticCell(it.first,.55f,it.second,.95f,3,.95f)}
        val points=ranges.flatMap {bounds -> (0..11).flatMap {y -> (0..11).map {x ->
            val u=bounds.first+(x+.5f)/12*(bounds.second-bounds.first);val v=.55f+(y+.5f)/12*.4f
            ArDepthPoint(u,v,3f-2*v,.95f,Vec3(u*2,-1.2f,-(3f-2*v)))
        } } }
        return pair.first.copy(cells=cells) to pair.second.copy(points=points)
    }
    @Test fun groundedCandidateRequiresThreeFreshObservationsAndNeverMeansPassable() {
        val e=ArSpatialEvaluator();assertNull(eval(e,groundSample(0)).candidateDirection);assertNull(eval(e,groundSample(1)).candidateDirection)
        val r=eval(e,groundSample(2));assertEquals(Direction.LEFT,r.candidateDirection)
        assertTrue(r.corridorReasons.contains("body_clearance_and_alignment_unverified"));assertNull(r.wallDistanceM)
    }
    @Test fun tiedUnknownOrNewObstacleRemovesCandidate() {
        val e=ArSpatialEvaluator();repeat(3) {assertNull(eval(e,groundSample(it,true)).candidateDirection)}
        repeat(3) {eval(e,groundSample(it+4))}
        val pair=groundSample(7)
        val obstacle=Detection(0,.9f,0f,.5f,.4f,1f)
        val r=e.evaluate(pair.first.key,pair.first,pair.second,listOf(obstacle),pair.first.key.capturedAtMs+150,true)
        assertNull(r.candidateDirection)
        assertNull(eval(e,pair.first to pair.second.copy(points=emptyList())).candidateDirection)
    }
    @Test fun corridorHistoryCannotSurviveSessionChangeOrLongGap() {
        val e=ArSpatialEvaluator();eval(e,groundSample(0));eval(e,groundSample(1))
        assertNull(eval(e,groundSample(8)).candidateDirection)
        val pair=groundSample(9);val s=pair.first.copy(key=pair.first.key.copy(generation=2))
        assertNull(eval(e,s to pair.second).candidateDirection)
    }
    @Test fun previewMappingPreservesCropAndLetterboxWithoutStretchingBoxes() {
        val m=ArPreviewMapping(listOf(-.2f,0f,1.2f,0f,-.2f,1f),.5f,1f)
        assertEquals(.5f,m.point(.5f,.5f).first,.0001f)
        assertEquals(.5f,m.point(.5f,.5f).second,.0001f)
        assertEquals(.15f,m.point(0f,0f).first,.0001f)
        assertEquals(.85f,m.point(1f,1f).first,.0001f)
    }
    @Test fun unknownCaptureClockCannotProduceMetricOrMotionSpeech() {
        val e=ArSpatialEvaluator();val p=sample(0)
        val r=eval(e,p.first to p.second.copy(captureClockKnown=false))
        assertNull(r.wallDistanceM);assertFalse(r.near);assertFalse(r.approaching)
        assertNull(r.candidateDirection);assertTrue("camera_clock_unknown" in r.reasons)
    }
}
