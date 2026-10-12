package org.sightguard.app

import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.CopyOnWriteArrayList

/** External fixtures are selected by GT before inference. This is still-image model validation, never CameraX motion acceptance. */
class FurnitureDatasetInstrumentedTest {
    @Test fun labelledRealSegFormerMasksAndFreshEvents() {
        val ins=InstrumentationRegistry.getInstrumentation();val context=ins.targetContext
        val root=File(context.filesDir,"furniture")
        val manifest=JSONObject(File(root,"manifest.json").readText());val cases=manifest.getJSONArray("cases")
        val arguments=InstrumentationRegistry.getArguments()
        val foreground=arguments.getString("foreground_fixture_test","false").toBooleanStrict()
        var fixtureActivity: MainActivity?=null
        val caseStart=arguments.getString("case_start","0").toInt()
        val caseCount=arguments.getString("case_count",cases.length().toString()).toInt()
        val modelSize=arguments.getString("semantic_size","256").toInt()
        require(caseStart in 0 until cases.length() && caseCount>0)
        val results=JSONArray();val audio=CopyOnWriteArrayList<SpeechPlaybackEvent>()
        val loadedConfig=loadEnvironmentConfig(context)
        val production=arguments.getString("production_config","false").toBooleanStrict()
        val config=loadedConfig.copy(automaticKinds=if(production) loadedConfig.automaticKinds else EnvironmentKind.entries.toSet(),
            automaticModelRequirements=if(production) loadedConfig.automaticModelRequirements else emptyMap(),
            useFamilyConfidence=arguments.getString("family_confidence",loadedConfig.useFamilyConfidence.toString()).toBooleanStrict())
        val engine=SemanticEnvironmentEngine(config);lateinit var speech: SpeechController
        ins.runOnMainSync {speech=SpeechController(context,{}, {}, {e->audio.add(e);if(e.phase==SpeechPlaybackPhase.ON_START) engine.playbackStarted(e.eventId!!)});speech.startLocalFallbackForTest()}
        val modelStarted=SystemClock.elapsedRealtime();var loadedMs=0L
        try {
            if(foreground) {
                fixtureActivity=ins.startActivitySync(android.content.Intent(context,MainActivity::class.java)
                    .putExtra("foreground_fixture_test",true)
                    .addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)) as MainActivity
                ins.runOnMainSync {
                    fixtureActivity!!.window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                    fixtureActivity!!.setContentView(android.widget.TextView(context).apply {
                        text="标注图片回放测试\n\n不是当前摄像头画面。\n测试提示不表示现场存在这些物体。\n完成后恢复正常感知。"
                        textSize=24f;setPadding(32,80,32,32)
                        setTextColor(android.graphics.Color.WHITE);setBackgroundColor(android.graphics.Color.BLACK)
                    })
                }
            }
            SpatialDetector(context,2,modelSize).use { detector ->
                loadedMs=SystemClock.elapsedRealtime()-modelStarted
                for(index in caseStart until minOf(cases.length(),caseStart+caseCount)) {
                    val case=cases.getJSONObject(index)
                    fun checked(name: String, hash: String): ByteArray {
                        val bytes=File(root,case.getString(name)).readBytes()
                        assertEquals(case.getString(hash),MessageDigest.getInstance("SHA-256").digest(bytes).joinToString(""){"%02x".format(it)})
                        return bytes
                    }
                    val bytes=checked("image","image_sha256");val maskBytes=checked("mask","mask_sha256")
                    val bitmap=BitmapFactory.decodeByteArray(bytes,0,bytes.size)
                    val annotation=BitmapFactory.decodeByteArray(maskBytes,0,maskBytes.size)
                    assertEquals(bitmap.width,annotation.width);assertEquals(bitmap.height,annotation.height)
                    val annotationPixels=IntArray(annotation.width*annotation.height)
                    annotation.getPixels(annotationPixels,0,annotation.width,0,0,annotation.width,annotation.height)
                    val histogram=IntArray(256)
                    annotationPixels.forEach {histogram[Color.red(it)]++}
                    for(kind in EnvironmentKind.entries.filter {it.furniture}) {
                        val fraction=kind.classIds.sumOf {histogram[it+1]}.toDouble()/annotationPixels.size
                        assertEquals("Android label decoding must preserve ADE20K annotation IDs",case.getJSONObject("gt_fractions").getDouble(kind.name),fraction,1e-9)
                    }
                    ins.runOnMainSync {engine.reset();speech.cancelAlerts("dataset_scene_changed")}
                    val observations=JSONArray();val caseEvents=mutableSetOf<String>();val supported=mutableSetOf<String>()
                    var statistics=JSONObject()
                    for(i in 0..3) {
                        val start=SystemClock.elapsedRealtime()
                        val evidence=detector.analyzeAll(bitmap,SpatialFrameKey(10000L+index,0,i,start.toDouble(),bitmap.width,bitmap.height,start))
                        val measureStart=SystemClock.elapsedRealtime();val parts=SemanticStructureMeasure(config).components(evidence.cells)
                        val measureMs=SystemClock.elapsedRealtime()-measureStart
                        supported.addAll(parts.filter {it.supported}.map {it.kind.name})
                        if(i==0) {
                            // All samples count, including abstained/low-confidence model cells (-1).
                            val prediction=IntArray(128*128){-1}
                            for(cell in evidence.cells) for(y in 0 until 128) {
                                if((y+.5f)/128 !in cell.top..<cell.bottom) continue
                                for(x in 0 until 128) if((x+.5f)/128 in cell.left..<cell.right) prediction[y*128+x]=cell.classId
                            }
                            statistics=JSONObject()
                            val maskImage=android.graphics.Bitmap.createBitmap(prediction.map {
                                Color.rgb(it+1,it+1,it+1)
                            }.toIntArray(),128,128,android.graphics.Bitmap.Config.ARGB_8888)
                            File(context.filesDir,"furniture-pred-${case.getString("image").removeSuffix(".jpg")}.png").outputStream().use {
                                maskImage.compress(android.graphics.Bitmap.CompressFormat.PNG,100,it)
                            }
                            maskImage.recycle()
                            for(kind in EnvironmentKind.entries.filter {it.furniture}) {
                                var tp=0;var fp=0;var fn=0;var tn=0
                                for(y in 0 until 128) for(x in 0 until 128) {
                                    val label=Color.red(annotationPixels[((y+.5f)/128*annotation.height).toInt()*annotation.width+((x+.5f)/128*annotation.width).toInt()])-1
                                    if(label<0) continue
                                    val gt=label in kind.classIds;val pred=prediction[y*128+x] in kind.classIds
                                    when {gt && pred->tp++;!gt && pred->fp++;gt->fn++;else->tn++}
                                }
                                statistics.put(kind.name,JSONObject().put("tp",tp).put("fp",fp).put("fn",fn).put("tn",tn))
                            }
                        }
                        var notices=emptyList<EnvironmentNotice>()
                        ins.runOnMainSync {
                            notices=engine.observe(evidence,SystemClock.elapsedRealtime(),true,false)
                            val desired=notices.firstOrNull {it.kind.furniture}
                            if(desired!=null && speech.canQueueOrdinaryAlert) {
                                caseEvents.add(desired.id)
                                speech.offer(desired.text,false,desired.id,desired.validUntilMs,desired.direction)
                            }
                            audio.filter {it.phase==SpeechPlaybackPhase.ON_START && it.eventId in caseEvents}.forEach {engine.playbackStarted(it.eventId!!)}
                        }
                        observations.put(JSONObject().put("latency_ms",evidence.latencyMs).put("source_age_ms",SystemClock.elapsedRealtime()-start)
                            .put("measure_ms",measureMs).put("pss_kb",android.os.Debug.getPss())
                            .put("structures",JSONArray(parts.map {environmentSupportJson(it)})).put("events",JSONArray(notices.map {it.id})))
                        SystemClock.sleep((650-(SystemClock.elapsedRealtime()-start)).coerceAtLeast(0))
                    }
                    val started=audio.filter {it.phase==SpeechPlaybackPhase.ON_START && it.eventId in caseEvents}
                    started.forEach {assertTrue(it.elapsedRealtimeMs<it.validUntilElapsedMs);assertFalse(SpeechPromptCatalog.resolve(it.spokenText!!).usedGeneric)}
                    assertEquals("same event must not repeat",started.map {it.eventId}.toSet().size,started.size)
                    results.put(JSONObject().put("case",case).put("pixels",statistics).put("supported",JSONArray(supported.toList())).put("observations",observations)
                        .put("audio_started",JSONArray(started.map {it.spokenText})))
                    bitmap.recycle();annotation.recycle()
                }
            }
        } finally {
            File(context.filesDir,"furniture-dataset-actual.json").writeText(JSONObject().put("schema",1).put("device",android.os.Build.MODEL)
                .put("sdk",android.os.Build.VERSION.SDK_INT).put("model_sha256",SegFormerModels.contract(modelSize).sha256).put("model_load_ms",loadedMs)
                .put("model_input_size",modelSize).put("app_version",BuildConfig.VERSION_NAME)
                .put("case_start",caseStart).put("case_count_requested",caseCount)
                .put("family_confidence",config.useFamilyConfidence)
                .put("evidence_level","real Android ONNX on labelled validation stills; not live camera or field accuracy")
                .put("metric","128x128 normalized pixel-center sampling; annotation 0 ignored; confidence-filtered model masks; missed cells count as FN")
                .put("all_kinds_enabled_for_test_only",!production).put("cases",results)
                .put("audio",JSONArray(audio.map {JSONObject().put("phase",it.phase.logName).put("event",it.eventId).put("text",it.spokenText).put("elapsed_ms",it.elapsedRealtimeMs).put("deadline_ms",it.validUntilElapsedMs)})).toString(2))
            ins.runOnMainSync {speech.close();fixtureActivity?.finish()}
        }
    }
}
