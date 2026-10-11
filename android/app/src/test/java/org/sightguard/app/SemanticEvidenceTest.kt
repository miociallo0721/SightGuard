package org.sightguard.app

import org.junit.Assert.*
import org.junit.Test

/** Synthetic rule fixtures; these tests do not validate real model accuracy. */
class SemanticEvidenceTest {
    private fun cell(id: Int, top: Float = .1f, bottom: Float = .3f, confidence: Float = .9f) = SemanticCell(.1f, top, .3f, bottom, id, confidence)
    @Test fun officialMappingAndTreeHeightRemainSceneSemantics() {
        assertEquals(setOf(3, 6, 11), SemanticEvidence.groundClassIds)
        val result = SemanticEvidence.summarize(listOf(cell(0), cell(14), cell(4), cell(4, .6f, .8f), cell(53), cell(59), cell(121)), emptyList(), SemanticSourceQuality())
        assertEquals(setOf(SemanticGroup.WALL, SemanticGroup.DOOR, SemanticGroup.TREE_UPPER, SemanticGroup.TREE_LOWER, SemanticGroup.STAIRS, SemanticGroup.STAIRWAY, SemanticGroup.STEP), result.regions.map { it.group }.toSet())
        assertFalse(result.overview.contains("安全")); assertFalse(result.overview.contains("障碍"))
    }
    @Test fun invalidLowConfidenceAndTinyEvidenceAbstain() {
        val cells = listOf(cell(0, confidence = .34f), cell(14, confidence = Float.NaN), SemanticCell(0f, 0f, .01f, .01f, 3, .9f), cell(999), cell(4, .8f, .2f))
        assertTrue(SemanticEvidence.summarize(cells, emptyList(), SemanticSourceQuality()).regions.isEmpty())
    }
    @Test fun aggregationIsAreaWeightedAndOrderIndependent() {
        val a = SemanticCell(.4f, 0f, .6f, .2f, 0, .4f)
        val b = SemanticCell(.3f, .2f, .7f, .4f, 0, 1f)
        val first = SemanticEvidence.summarize(listOf(a, b), emptyList(), SemanticSourceQuality())
        assertEquals(first, SemanticEvidence.summarize(listOf(b, a), emptyList(), SemanticSourceQuality()))
        assertEquals(.12f, first.regions.single().areaFraction, .00001f)
        assertEquals(.8f, first.regions.single().confidence, .00001f)
    }
    @Test fun staleAndUnreadableSourcesSuppressBothInputs() {
        val person = Detection(0, .9f, .1f, .1f, .3f, .8f)
        for (quality in listOf(SemanticSourceQuality(current = false), SemanticSourceQuality(readable = false), SemanticSourceQuality(known = false, detail = "UNKNOWN"))) {
            val summary = SemanticEvidence.summarize(listOf(cell(3)), listOf(person), quality)
            assertTrue(summary.regions.isEmpty()); assertFalse(summary.overview.contains("行人"))
            assertEquals("当前环境信息不足。", summary.overview)
            assertTrue(summary.reason.isNotEmpty())
        }
        assertTrue(SemanticEvidence.summarize(emptyList(), listOf(person, person), SemanticSourceQuality()).overview.contains("检测到左侧行人"))
    }
    @Test fun staticClassesKeepIndividualRawIdsAndPosition() {
        val ids = listOf(10, 15, 19, 32, 93)
        val summary = SemanticEvidence.summarize(ids.map { cell(it) }, emptyList(), SemanticSourceQuality())
        assertEquals(ids.toSet(), summary.regions.map { it.classId }.toSet())
        assertTrue(summary.regions.all { it.group == SemanticGroup.STATIC && it.direction == Direction.LEFT })
        for (name in listOf("柜子", "桌子", "椅子", "围栏", "杆柱")) assertTrue(summary.overview.contains("左侧$name"))
    }
    @Test fun separateDirectionsAndDetectorPositionsAreDeterministic() {
        val cells = listOf(SemanticCell(.1f, .1f, .2f, .3f, 14, .9f), SemanticCell(.45f, .1f, .55f, .3f, 14, .9f), SemanticCell(.8f, .1f, .9f, .3f, 14, .9f))
        val people = cells.map { Detection(0, .9f, it.left, it.top, it.right, it.bottom) }
        val result = SemanticEvidence.summarize(cells, people, SemanticSourceQuality())
        assertEquals(setOf(Direction.LEFT, Direction.CENTER, Direction.RIGHT), result.regions.map { it.direction }.toSet())
        for (position in listOf("左侧", "前方", "右侧")) {
            assertTrue(result.overview.contains(position + "门")); assertTrue(result.overview.contains(position + "行人"))
        }
        assertEquals(result, SemanticEvidence.summarize(cells.reversed(), people.reversed(), SemanticSourceQuality()))
        assertEquals("当前环境信息不足。", SemanticEvidence.summarize(emptyList(), emptyList(), SemanticSourceQuality()).overview)
    }
}
