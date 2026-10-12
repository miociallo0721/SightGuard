package org.sightguard.app

import org.junit.Assert.*
import org.junit.Test

class LocalSpacePreviewTest {
    private val key=SpatialFrameKey(7,2,3,100.0,641,479,100)
    private val cell=GridCell(14,7,.46f,.93f,.53f,1f,1f,0f,0f,0f,null,"GROUND","VALID",emptyList(),emptyMap())
    @Test fun clipsActualGroundAndNeverFillsUnknownHoles() {
        val good=SemanticCell(.47f,.94f,.49f,.96f,3,.9f)
        val other=good.copy(left=.50f,right=.52f)
        val clipped=LocalSpacePreview.clip(listOf(good,other,good.copy(classId=0),good.copy(confidence=.4f),good.copy(left=Float.NaN)),listOf(cell))
        assertEquals(listOf(good,other),clipped)
        assertTrue(clipped.none {it.left<.50f && it.right>.49f})
        assertTrue(LocalSpacePreview.clip(listOf(good),emptyList()).isEmpty())
    }
    @Test fun boundaryClippingHandlesOddPixelGridAndDoesNotExtendEvidence() {
        val s=SemanticCell(.45f,.92f,.55f,1f,3,.9f)
        val clipped=LocalSpacePreview.clip(listOf(s),listOf(cell)).single()
        assertEquals(cell.left,clipped.left);assertEquals(cell.right,clipped.right)
        assertEquals(cell.top,clipped.top);assertEquals(1f,clipped.bottom)
    }
    @Test fun evenSmallDetectionOcclusionSuppressesPreview() {
        val r=LocalSpaceRecord(key,"SUPPORTED_CANDIDATE","test",supportedCells=listOf(cell))
        assertTrue(LocalSpacePreview.obstructed(r,listOf(Detection(0,.9f,.48f,.95f,.481f,.951f))))
        assertFalse(LocalSpacePreview.obstructed(r,listOf(Detection(0,.9f,.1f,.1f,.2f,.2f))))
        assertTrue(LocalSpacePreview.clip(listOf(SemanticCell(.47f,.94f,.49f,.96f,3,.9f)),listOf(cell.copy(occupancy=.001f))).isEmpty())
    }
    @Test fun expiresAndRejectsMotionSourceGeometryAndUnconfirmedStates() {
        val r=LocalSpaceRecord(key,"SUPPORTED_CANDIDATE","test",validUntilMs=1300,previewCells=listOf(SemanticCell(.47f,.94f,.49f,.96f,3,.9f)))
        fun visible(x: LocalSpaceRecord=r,now: Long=200,g: Long=7,e: Long=2,w: Int=641,rapid: Boolean=false)=LocalSpacePreview.visible(x,now,g,e,w,479,rapid)
        assertTrue(visible());assertTrue(visible(now=1299));assertFalse(visible(now=1300));assertFalse(visible(now=99))
        assertFalse(visible(g=8));assertFalse(visible(e=3));assertFalse(visible(w=479));assertFalse(visible(rapid=true))
        for(status in listOf("CONFIRMING","TIED","ABSTAIN","INSUFFICIENT_EVIDENCE")) assertFalse(visible(r.copy(status=status)))
        assertFalse(visible(r.copy(previewCells=emptyList())))
    }
}
