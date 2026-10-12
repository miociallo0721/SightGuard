package org.sightguard.app

import org.junit.Assert.*
import org.junit.Test

class DetectorTaxonomyTest {
    @Test fun originalIdentityAndNewLabelsRemainCompatible() {
        assertEquals((0..79).toList(),DetectorTaxonomy.yoloeMapping.take(80))
        assertEquals(90,DetectorTaxonomy.yoloeMapping.size)
        assertTrue(DetectorTaxonomy.yoloeMapping.all {it in Labels.detectionNames.indices})
        assertNull(DetectorTaxonomy.canonical(-1));assertNull(DetectorTaxonomy.canonical(90))
        assertEquals("显示器",Labels.detectionNames[DetectorTaxonomy.canonical(89)!!])
    }
    @Test fun synonymsShareIdentityAndUnsupportedFurnitureStaysContextual() {
        assertEquals(60,DetectorTaxonomy.canonical(80));assertEquals(60,DetectorTaxonomy.canonical(81))
        assertTrue((82..84).all {DetectorTaxonomy.canonical(it)==56})
        assertEquals(DetectorTaxonomy.canonical(85),DetectorTaxonomy.canonical(86))
        val config=RelevanceConfig()
        assertTrue((85..89).map {DetectorTaxonomy.canonical(it)}.none {it in config.primaryClasses || it in config.conditionalClasses})
    }
}
