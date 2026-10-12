package org.sightguard.app

/** Keep COCO identities stable; synonyms share existing tracking/risk identity. */
object DetectorTaxonomy {
    val yoloeMapping: List<Int> = (0..79).toList() + listOf(60,60,56,56,56,80,80,81,82,83)
    fun canonical(sourceId: Int): Int? = yoloeMapping.getOrNull(sourceId)
}
