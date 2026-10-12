package org.sightguard.app

import org.junit.Assert.*
import org.junit.Test

class FurnitureSemanticsTest {
    private fun enabledChair() = SemanticEnvironmentEngine(EnvironmentConfig(automaticKinds=setOf(EnvironmentKind.CHAIR)))
    @Test fun unvalidatedFurnitureIsNotEnabledByDefault() {
        assertFalse(EnvironmentConfig().automaticKinds.any { it.furniture })
    }
    @Test fun modelFallbackCannotInheritFurnitureApprovalOrMixedModelConfirmation() {
        val approved="a".repeat(64)
        val engine=SemanticEnvironmentEngine(EnvironmentConfig(automaticKinds=setOf(EnvironmentKind.BED),
            automaticModelRequirements=mapOf(EnvironmentKind.BED to approved)))
        fun frame(t:Long,sha:String)=evidence(t,mask(7)).copy(modelSha=sha)
        assertTrue(engine.observe(frame(1000,approved),1100,true,false).isEmpty())
        assertTrue(engine.observe(frame(1700,SpatialDetector.MODEL_SHA),1800,true,false).isEmpty())
        assertTrue(engine.observe(frame(2400,approved),2500,true,false).isEmpty())
        val notice=engine.observe(frame(3100,approved),3200,true,false).single()
        engine.playbackStarted(notice.id)
        assertTrue(engine.observe(frame(3300,SpatialDetector.MODEL_SHA),3400,true,false).isEmpty())
        assertTrue(engine.currentNotices(3400).isEmpty())
    }
    @Test fun resolutionFallbackIsBoundedOneWayAndDoesNotUseSingleSlowSample() {
        val policy=SemanticResolutionPolicy()
        assertFalse(policy.observe(410,0));assertFalse(policy.observe(200,0))
        assertFalse(policy.observe(410,0));assertTrue(policy.observe(410,0))
        assertTrue(policy.observe(100,0))
        assertTrue(SemanticResolutionPolicy().observe(100,5))
        assertTrue(SemanticResolutionPolicy().observe(100,0,true))
        val configured=SemanticResolutionPolicy(500,3,8)
        repeat(3) {assertFalse(configured.observe(450,5))}
        assertFalse(configured.observe(600,0));assertFalse(configured.observe(600,0))
        assertTrue(configured.observe(600,0))
    }
    private fun mask(id: Int) = (0 until 20).flatMap { y -> (0 until 20).map { x ->
        SemanticCell(x/20f,y/20f,(x+1)/20f,(y+1)/20f,id,.9f)
    } }
    private fun evidence(time: Long, cells: List<SemanticCell>) = GroundEvidence(
        SpatialFrameKey(1,0,time.toInt(),time.toDouble(),640,480,time),cells,100,time+100,"real-contract-test-fixture")
    @Test fun everyFurnitureRawClassMapsToOneFamilyAndConfirmsWithoutDepth() {
        for(label in EnvironmentTaxonomy.furnitureLabels) {
            val kind=EnvironmentTaxonomy.kindForClass(label.id)!!
            assertTrue(kind.furniture)
            val engine=SemanticEnvironmentEngine(EnvironmentConfig(automaticKinds=setOf(kind), thresholds=mapOf(kind to StructureThreshold(.035f,12,.6f,3))))
            for(t in listOf(1000L,1700L)) assertTrue(engine.observe(evidence(t,mask(label.id)),t+100,true,false).isEmpty())
            val notice=engine.observe(evidence(2400,mask(label.id)),2500,true,false).single()
            assertEquals(kind,notice.kind); assertEquals(setOf(label.id),notice.rawClassIds)
            engine.playbackStarted(notice.id)
            assertTrue(engine.observe(evidence(3100,mask(label.id)),3200,true,false).isEmpty())
            assertTrue(engine.currentNotices(8000).isEmpty())
        }
    }
    @Test fun adjacentChairSubclassesAggregateButTwoDisconnectedChairsRemainSeparate() {
        val mixed=mask(19).map {if(it.left>.5f) it.copy(classId=75) else it}
        val measure=SemanticStructureMeasure(EnvironmentConfig())
        assertEquals(setOf(19,75),measure.components(mixed).single().rawClassIds)
        assertEquals(2,measure.components(mixed.filter {it.right<=.4f || it.left>=.6f}).size)
    }
    @Test fun familyPosteriorIsAnExplicitUnionWithRawConfidencePreservedAndNoThresholdReduction() {
        val cells=mask(19).map {it.copy(confidence=.4f,familyConfidence=.75f)}
        val raw=SemanticStructureMeasure(EnvironmentConfig()).components(cells)
        assertTrue(raw.isEmpty())
        val union=SemanticStructureMeasure(EnvironmentConfig(useFamilyConfidence=true)).components(cells).single()
        assertTrue(union.supported);assertEquals(.4f,union.rawConfidence,.00001f)
        assertEquals(.75f,union.confidence,.00001f);assertEquals("furniture_family_softmax_sum",union.confidenceBasis)
        for(invalid in listOf(Float.NaN,.3f,1.1f)) assertTrue(SemanticStructureMeasure(EnvironmentConfig(useFamilyConfidence=true))
            .components(cells.map {it.copy(familyConfidence=invalid)}).isEmpty())
        assertTrue(SemanticStructureMeasure(EnvironmentConfig(useFamilyConfidence=true))
            .components(cells.map {it.copy(classId=0)}).isEmpty())
    }
    @Test fun lowConfidenceStaleAndUnverifiedFurnitureNeverCreatesAutomaticNotice() {
        val engine=SemanticEnvironmentEngine(EnvironmentConfig(automaticKinds=setOf(EnvironmentKind.WALL)))
        for(t in listOf(1000L,1700L,2400L)) assertTrue(engine.observe(evidence(t,mask(19)),t+100,true,false).isEmpty())
        assertTrue(engine.silenceReason.contains("尚未验证"))
        val enabled=enabledChair()
        assertTrue(enabled.observe(evidence(1000,mask(19)),1600,true,false).isEmpty())
        assertTrue(enabled.observe(evidence(1700,mask(19).map {it.copy(confidence=.3f)}),1800,true,false).isEmpty())
    }
    @Test fun impossibleCompletionAndNegativeModelLatencyCannotConfirmFurniture() {
        for(invalid in listOf(evidence(1700,mask(19)).copy(completedAtMs=2000),evidence(1700,mask(19)).copy(latencyMs=-1))) {
            val engine=enabledChair();engine.observe(evidence(1000,mask(19)),1100,true,false)
            assertTrue(engine.observe(invalid,1800,true,false).isEmpty())
            assertTrue(engine.currentNotices(1800).isEmpty())
        }
    }
    @Test fun crossSourceDedupRequiresActualStartedCompatibleNearbyRegionAndSameSession() {
        val dedup=EnvironmentSpeechDeduplicator()
        val support=SemanticStructureMeasure(EnvironmentConfig()).components(mask(19)).single()
        val notice=EnvironmentNotice("chair",EnvironmentKind.CHAIR,1000.0,1,1500,support=support)
        assertFalse(dedup.alreadyHeard(1,0,notice,1100))
        dedup.yoloStarted(1,0,56,listOf(0.0,0.0,1.0,1.0),1050,"前方有目标")
        assertFalse(dedup.alreadyHeard(1,0,notice,1100))
        dedup.yoloStarted(1,0,56,listOf(0.0,0.0,1.0,1.0),1050,"前方有椅子")
        assertTrue(dedup.alreadyHeard(1,0,notice,1100))
        assertFalse(dedup.alreadyHeard(2,0,notice,1100))
        assertFalse(dedup.alreadyHeard(1,1,notice,1100))
        assertFalse(dedup.alreadyHeard(1,0,notice.copy(kind=EnvironmentKind.TABLE),1100))
        assertFalse(dedup.alreadyHeard(1,0,notice,1500))
        assertFalse(dedup.alreadyHeard(1,0,notice.copy(validUntilMs=10000),6000))
        dedup.reset();dedup.yoloStarted(1,0,56,listOf(0.0,0.0,.2,.2),1050,"前方有椅子")
        assertFalse(dedup.alreadyHeard(1,0,notice,1100))
    }
    @Test fun overviewGroupsChairSubclassesAndYoloWithoutClaimingObjectCount() {
        val cells=listOf(SemanticCell(.4f,.3f,.5f,.5f,19,.9f),SemanticCell(.5f,.3f,.6f,.5f,75,.9f))
        val summary=SemanticEvidence.summarize(cells,listOf(Detection(56,.9f,.4f,.3f,.6f,.5f)),SemanticSourceQuality())
        assertEquals(setOf(19,75),summary.regions.map {it.classId}.toSet())
        assertEquals(1,Regex("椅子").findAll(summary.overview).count())
    }
    @Test fun detectionAcknowledgmentDoesNotPretendSemanticPlaybackOrRepeatPersistentPresence() {
        val engine=enabledChair();engine.observe(evidence(1000,mask(19)),1100,true,false)
        val notice=engine.observe(evidence(1700,mask(19)),1800,true,false).single()
        engine.acknowledgedByDetection(notice.id)
        assertTrue(engine.observe(evidence(2400,mask(19)),2500,true,false).isEmpty())
        assertEquals("同一区域已由目标提醒播报。",engine.silenceReason)
    }
    @Test fun furnitureGridIsOptInAndDoesNotInventFreeSpaceWithMissingSources() {
        val key=evidence(1000,emptyList()).key
        val source=GridSemanticSource(key,mask(19),"fixture")
        val motion=MotionEvidence(MotionQuality.UNKNOWN,"not synchronized")
        assertTrue(SpatialGridFusion().fuse(key,source,null,null,motion).cells.all {it.state=="UNKNOWN"})
        val result=SpatialGridFusion(GridConfig(includeFurniture=true)).fuse(key,source,null,null,motion)
        assertTrue(result.cells.all {it.state=="OBJECT" && it.quality=="LIMITED" && it.semanticObjectClassIds==setOf(19)})
        assertFalse(result.productionDecisionChanged)
        assertTrue(SpatialGridFusion(GridConfig(includeFurniture=true)).fuse(key,source.copy(key=key.copy(epoch=1)),null,null,motion).cells.all {it.state=="UNKNOWN"})
    }
}
