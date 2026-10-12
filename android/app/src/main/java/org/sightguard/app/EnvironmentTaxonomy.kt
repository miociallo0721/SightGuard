package org.sightguard.app

/** ADE20K logit IDs, never annotation IDs (annotation zero is ignored). */
object EnvironmentTaxonomy {
    const val VERSION = "sightguard.environment-taxonomy.v1"
    data class Label(val id: Int, val rawName: String, val chineseName: String)
    val furnitureLabels: List<Label> = listOf(
        Label(7,"bed ","床"), Label(10,"cabinet","柜子"), Label(15,"table","桌子"),
        Label(19,"chair","椅子"), Label(23,"sofa","沙发"), Label(24,"shelf","架子"),
        Label(30,"armchair","扶手椅"), Label(31,"seat","座椅"), Label(33,"desk","书桌"),
        Label(35,"wardrobe","衣柜"), Label(45,"counter","柜台"), Label(50,"refrigerator","冰箱"),
        Label(64,"coffee table","茶几"), Label(69,"bench","长椅"), Label(70,"countertop","台面"),
        Label(75,"swivel chair","转椅"), Label(110,"stool","凳子"),
    )
    val furnitureClassIds: Set<Int> = furnitureLabels.map {it.id}.toSet()
    val rawContract: Map<Int,String> = furnitureLabels.associate {it.id to it.rawName}
    private val names = furnitureLabels.associate {it.id to it.chineseName}
    fun chineseName(id: Int): String? = names[id]
    fun kindForClass(id: Int): EnvironmentKind? = EnvironmentKind.entries.firstOrNull {id in it.classIds}
    fun family(id: Int): EnvironmentKind? = when(id) {
        53,59,121 -> EnvironmentKind.STAIRS
        else -> kindForClass(id)
    }
    fun yoloKind(classId: Int): EnvironmentKind? = when(classId) {
        13,56 -> EnvironmentKind.CHAIR
        57 -> EnvironmentKind.SOFA
        59 -> EnvironmentKind.BED
        60 -> EnvironmentKind.TABLE
        72 -> EnvironmentKind.APPLIANCE
        else -> null
    }
    val furnitureFamilyIds: Map<Int,Set<Int>> = furnitureClassIds.associateWith {kindForClass(it)!!.classIds}
}
