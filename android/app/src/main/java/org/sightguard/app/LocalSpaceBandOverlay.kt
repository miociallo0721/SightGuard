package org.sightguard.app

import android.os.SystemClock
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.withTransform
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay

/** Shared FIT_CENTER coordinates; retain semantic holes, never interpolate a route. */
@Composable internal fun LocalSpaceBandOverlay(record: LocalSpaceRecord) {
    var expired by remember(record.key,record.validUntilMs) { mutableStateOf(SystemClock.elapsedRealtime()>=record.validUntilMs) }
    LaunchedEffect(record.key,record.validUntilMs) {
        delay((record.validUntilMs-SystemClock.elapsedRealtime()).coerceAtLeast(0))
        expired=true
    }
    if(expired || record.status!="SUPPORTED_CANDIDATE" || record.previewCells.isEmpty()) return
    // One cached path per observation, no per-frame mask allocation or new inference.
    val path=remember(record) {Path().apply { record.previewCells.forEach {addRect(Rect(it.left,it.top,it.right,it.bottom))} }}
    Box(Modifier.fillMaxSize()) {
        Canvas(Modifier.fillMaxSize().semantics {contentDescription="局部空间候选，通行未确认"}) {
            if(SystemClock.elapsedRealtime()>=record.validUntilMs) return@Canvas
            val fit=fitImage(record.key.width,record.key.height,size.width,size.height)
            withTransform({
                translate(fit.offsetX,fit.offsetY)
                scale(record.key.width*fit.scale,record.key.height*fit.scale,androidx.compose.ui.geometry.Offset.Zero)
            }) { drawPath(path,Color(0xFF9DB7E8).copy(alpha=.32f)) }
        }
        Text("空间候选 · 通行未确认",Modifier.padding(8.dp).background(Color(0xDD18202D)).padding(6.dp),color=Color.White,fontSize=12.sp)
    }
}
