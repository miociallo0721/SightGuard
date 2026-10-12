package org.sightguard.app

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File

/** Real phone renderer + actual ONNX photograph. This is not live-camera route validation. */
class LocalSpacePreviewInstrumentedTest {
    @Test fun actualPublicModelBandRendersThenExpires() {
        val ins=InstrumentationRegistry.getInstrumentation();val context=ins.targetContext
        val prefs=context.getSharedPreferences("perception",0);val auto=prefs.getBoolean("auto_start",true)
        prefs.edit().putBoolean("auto_start",false).commit()
        var activity: MainActivity?=null
        var bitmap: Bitmap?=null
        var record: LocalSpaceRecord?=null
        val started=SystemClock.elapsedRealtime()
        try {
            val host=ins.startActivitySync(Intent(context,MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) as MainActivity
            activity=host
            val source=ins.context.assets.open("local_space/ADE_val_00000065.jpg").use {BitmapFactory.decodeStream(it)}
            bitmap=source
            Detector(context,true).use {yolo -> SpatialDetector(context,1,256).use {semantic ->
                val engine=LocalSpaceCandidateEvaluator(loadLocalSpaceConfig(context))
                for(i in 1..3) {
                    val now=SystemClock.elapsedRealtime();val key=SpatialFrameKey(1,0,i,now.toDouble(),source.width,source.height,now)
                    val d=yolo.detect(source);val sem=semantic.analyzeAll(source,key)
                    val grid=SpatialGridFusion(GridConfig(minConfidence=.65f,includeFurniture=true)).fuse(key,GridSemanticSource(key,sem.cells,sem.modelSha),null,GridDetectionSource(key,d,"actual-yoloe"),MotionEvidence(MotionQuality.UNKNOWN,"repeated public photograph"))
                    record=engine.observe(grid,sem,SystemClock.elapsedRealtime(),true)
                }
                val r=record!!
                File(context.filesDir,"preview-band-validation.json").writeText(localSpaceJson(r).put("device",android.os.Build.MODEL).put("preview_rectangles",r.previewCells.size).put("source_age_ms",SystemClock.elapsedRealtime()-r.key.capturedAtMs).put("pss_kb",android.os.Debug.getPss()).put("scope","actual models on public static photo; phone UI rendering, not live route").toString(2))
                assertEquals("SUPPORTED_CANDIDATE",r.status);assertTrue(r.previewCells.isNotEmpty())
                ins.runOnMainSync {host.setContent {Box(Modifier.fillMaxSize().background(Color.Black)) {
                    Image(source.asImageBitmap(),null,Modifier.fillMaxSize(),contentScale=ContentScale.Fit)
                    LocalSpaceBandOverlay(r)
                }}}
                ins.waitForIdleSync();SystemClock.sleep(120)
                val before=ins.uiAutomation.takeScreenshot();assertNotNull(before)
                File(context.filesDir,"preview-band-visible.png").outputStream().use {before!!.compress(Bitmap.CompressFormat.PNG,100,it)}
                assertTrue("Render must occur before evidence expires",SystemClock.elapsedRealtime()<r.validUntilMs)
                SystemClock.sleep((r.validUntilMs-SystemClock.elapsedRealtime()).coerceAtLeast(0)+150)
                ins.waitForIdleSync()
                val after=ins.uiAutomation.takeScreenshot()!!
                File(context.filesDir,"preview-band-expired.png").outputStream().use {after.compress(Bitmap.CompressFormat.PNG,100,it)}
                var changed=0
                for(y in before!!.height/2 until before.height-100 step 2) for(x in 0 until before.width step 2)
                    if(before.getPixel(x,y)!=after.getPixel(x,y)) changed++
                assertTrue("Candidate pixels must disappear after deadline",changed>100)
                val file=File(context.filesDir,"preview-band-validation.json")
                file.writeText(JSONObject(file.readText()).put("expired_changed_pixels_sampled",changed).put("elapsed_ms",SystemClock.elapsedRealtime()-started).put("expiry_render_passed",true).toString(2))
                before.recycle();after.recycle()
            }}
        } finally {
            // Restore even if startActivitySync fails before the activity is returned.
            prefs.edit().putBoolean("auto_start",auto).commit()
            ins.runOnMainSync {activity?.finish()};ins.waitForIdleSync();bitmap?.recycle()
        }
    }
}
