package org.sightguard.app

import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.*
import androidx.compose.ui.unit.dp

/** Ordinary mode contains no algorithm controls; visual and spoken states have separate identities. */
@Composable
fun AccessibleHome(
    perceptionState: String, currentAlert: String, silenceReason: String, recentSpeech: String,
    description: String?, running: Boolean, muted: Boolean,
    onDescribe: () -> Unit, onMute: () -> Unit, onToggle: () -> Unit, onSettings: () -> Unit,
    preview: @Composable () -> Unit,
) {
    Surface(Modifier.fillMaxSize()) {
        BoxWithConstraints(Modifier.fillMaxSize().safeDrawingPadding().padding(12.dp).semantics { isTraversalGroup=true }) {
            val largeFont=LocalDensity.current.fontScale>1.4f
            val landscape=maxWidth>maxHeight && maxWidth>520.dp
            val stackActions=largeFont || maxWidth<330.dp
            val previewHeight=if(largeFont) 130.dp else (maxHeight*.33f).coerceIn(140.dp,320.dp)
            val info: @Composable () -> Unit = {
                Text("SightGuard · 视界",style=MaterialTheme.typography.titleLarge,modifier=Modifier.semantics { heading(); traversalIndex=0f })
                Text(perceptionState,style=MaterialTheme.typography.titleMedium,modifier=Modifier.semantics { traversalIndex=1f })
            }
            val controls: @Composable () -> Unit = {
                Card(Modifier.fillMaxWidth().semantics(mergeDescendants=true) { traversalIndex=3f }) {
                    Column(Modifier.padding(12.dp),verticalArrangement=Arrangement.spacedBy(6.dp)) {
                        Text("当前有效提醒",style=MaterialTheme.typography.labelLarge)
                        Text(currentAlert,style=MaterialTheme.typography.titleMedium)
                        Text(silenceReason,style=MaterialTheme.typography.bodyMedium)
                    }
                }
                // Deliberately no liveRegion: changing evidence must not continually interrupt TalkBack.
                Text("最近播报：$recentSpeech",style=MaterialTheme.typography.bodyMedium,modifier=Modifier.semantics { traversalIndex=4f })
                if(description!=null) Text(description,style=MaterialTheme.typography.bodyLarge,modifier=Modifier.semantics { traversalIndex=5f })
                val describe: @Composable (Modifier) -> Unit = { modifier ->
                    Button(onDescribe,modifier.heightIn(min=56.dp).semantics { traversalIndex=6f }) { Text("描述周围") }
                }
                val mute: @Composable (Modifier) -> Unit = { modifier ->
                    OutlinedButton(onMute,modifier.heightIn(min=56.dp).semantics { traversalIndex=7f; stateDescription=if(muted) "已静音" else "提醒已开启" }) { Text(if(muted) "恢复提醒" else "静音提醒") }
                }
                val stop: @Composable (Modifier) -> Unit = { modifier ->
                    Button(onToggle,modifier.heightIn(min=56.dp).semantics { traversalIndex=8f }) { Text(if(running) "停止感知" else "开始感知") }
                }
                val settings: @Composable (Modifier) -> Unit = { modifier ->
                    OutlinedButton(onSettings,modifier.heightIn(min=56.dp).semantics { traversalIndex=9f }) { Text("设置") }
                }
                if(stackActions) {
                    describe(Modifier.fillMaxWidth()); mute(Modifier.fillMaxWidth()); stop(Modifier.fillMaxWidth()); settings(Modifier.fillMaxWidth())
                } else {
                    Row(horizontalArrangement=Arrangement.spacedBy(8.dp)) { describe(Modifier.weight(1f)); mute(Modifier.weight(1f)) }
                    Row(horizontalArrangement=Arrangement.spacedBy(8.dp)) { stop(Modifier.weight(1f)); settings(Modifier.weight(1f)) }
                }
            }
            val previewModifier=Modifier.clip(RoundedCornerShape(12.dp)).background(Color.Black)
                .clearAndSetSemantics { contentDescription="摄像头预览，检测结果请使用描述周围"; traversalIndex=2f }
            if(landscape) Row(horizontalArrangement=Arrangement.spacedBy(12.dp)) {
                Box(previewModifier.weight(1f).fillMaxHeight()) { preview() }
                Column(Modifier.weight(1f).verticalScroll(rememberScrollState()),verticalArrangement=Arrangement.spacedBy(9.dp)) { info(); controls() }
            } else Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()),verticalArrangement=Arrangement.spacedBy(8.dp)) {
                info()
                Box(previewModifier.fillMaxWidth().height(previewHeight)) { preview() }
                controls()
            }
        }
    }
}
