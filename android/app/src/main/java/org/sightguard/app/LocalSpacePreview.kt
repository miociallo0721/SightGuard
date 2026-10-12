package org.sightguard.app

/** Image evidence only: never fill unknown holes or smooth across obstacle boundaries. */
object LocalSpacePreview {
    fun visible(record: LocalSpaceRecord, now: Long, generation: Long, epoch: Long,
        width: Int, height: Int, rapid: Boolean): Boolean =
        record.status == "SUPPORTED_CANDIDATE" && record.previewCells.isNotEmpty() &&
        now >= record.key.capturedAtMs && now < record.validUntilMs && !rapid &&
        record.key.generation == generation && record.key.epoch == epoch &&
        record.key.width == width && record.key.height == height

    /** Rendering is stricter than event scoring: even a small new box hides the band. */
    fun obstructed(record: LocalSpaceRecord, detections: List<Detection>): Boolean =
        record.supportedCells.any {c -> detections.any {d ->
            d.confidence.isFinite() && d.confidence>=Detector.MIN_SCORE &&
            d.left<c.right && d.right>c.left && d.top<c.bottom && d.bottom>c.top
        }}

    /** Index the 15x15 grid once; work is proportional to semantic pixels, not pixels x grid. */
    fun clip(semantics: List<SemanticCell>, supported: List<GridCell>): List<SemanticCell> {
        val index=supported.filter {it.occupancy == 0f}.associateBy {it.row*15+it.col}
        return buildList {
            for(s in semantics) {
                if(s.classId !in SpatialDetector.GROUND_CLASSES || !s.confidence.isFinite() || s.confidence<.65f ||
                    !listOf(s.left,s.top,s.right,s.bottom).all {it.isFinite() && it in 0f..1f} ||
                    s.right<=s.left || s.bottom<=s.top) continue
                for(r in ((s.top*15).toInt()-1).coerceIn(0,14)..((s.bottom*15).toInt()+1).coerceIn(0,14))
                    for(c in ((s.left*15).toInt()-1).coerceIn(0,14)..((s.right*15).toInt()+1).coerceIn(0,14)) {
                        val g=index[r*15+c] ?: continue
                        val l=maxOf(s.left,g.left); val t=maxOf(s.top,g.top)
                        val right=minOf(s.right,g.right); val b=minOf(s.bottom,g.bottom)
                        if(right>l && b>t) add(s.copy(left=l,top=t,right=right,bottom=b))
                    }
            }
        }
    }
}
