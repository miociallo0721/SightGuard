package org.sightguard.app

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.sightguard.app.core.*

class AssessmentReplayTest {
    private fun fixture(): JSONObject {
        val resource = javaClass.classLoader!!.getResourceAsStream("parity_reference.json")
        if (resource != null) return JSONObject(resource.bufferedReader().use { it.readText() })
        // On-device runner has Android assets rather than JVM classpath resources.
        val registry = Class.forName("androidx.test.platform.app.InstrumentationRegistry")
        val instrument = registry.getMethod("getInstrumentation").invoke(null)
        val context = Class.forName("android.app.Instrumentation").getMethod("getContext").invoke(instrument)
        val assets = Class.forName("android.content.Context").getMethod("getAssets").invoke(context)
        val stream = Class.forName("android.content.res.AssetManager").getMethod("open", String::class.java).invoke(assets, "parity_reference.json") as java.io.InputStream
        return JSONObject(stream.bufferedReader().use { it.readText() })
    }
    @Test fun realCameraTraceMatchesPcScoresEventsAlertsAndTimes() { replay(fixture().getJSONObject("realCameraTrace")) }
    @Test fun syntheticLifecycleScenariosMatchPcWithoutClaimingModelEvidence() {
        val scenarios = fixture().getJSONArray("syntheticScenarios")
        for (i in 0 until scenarios.length()) replay(scenarios.getJSONObject(i))
    }
    private fun replay(scenario: JSONObject) {
        val reference = scenario.getJSONObject("reference")
        val expected = reference.getJSONArray("steps")
        val streamId = expected.getJSONObject(0).getJSONObject("frame").getString("streamId")
        val engine = AssessmentEngine(streamId = streamId)
        val inputs = scenario.getJSONArray("frames")
        assertEquals(inputs.length(), expected.length())
        for (i in 0 until inputs.length()) compare(expected.getJSONObject(i), engine.step(parse(inputs.getJSONObject(i))), "$streamId.frame$i")
        compare(reference.getJSONArray("finishChanges"), engine.finish(), "$streamId.finish")
        assertTrue(engine.finish().isEmpty())
    }
    private fun parse(f: JSONObject): PFrame {
        val ds = f.getJSONArray("detections")
        return PFrame(f.getInt("frameIndex"), f.getDouble("timestampMs"), f.getInt("frameWidth"), f.getInt("frameHeight"),
            (0 until ds.length()).map { i -> val d = ds.getJSONObject(i); val b = d.getJSONArray("bboxXyxy")
                PDetection(if (d.isNull("trackId")) null else d.getLong("trackId"), d.getInt("classId"), d.getString("className"), d.getDouble("confidence"), (0..3).map(b::getDouble)) },
            f.getString("status"), if (f.isNull("error")) null else f.getString("error"), f.getString("timestampSource"))
    }
    private fun compare(expected: Any?, actual: Any?, path: String) {
        when {
            expected == null || expected == JSONObject.NULL -> assertNull(path, actual)
            expected is Number -> { assertTrue("$path is not numeric: $actual", actual is Number); assertEquals(path, expected.toDouble(), (actual as Number).toDouble(), 1e-9) }
            expected is JSONArray -> {
                val list = actual as? List<*> ?: error("$path is not a list")
                assertEquals(path, expected.length(), list.size)
                for (i in list.indices) compare(expected.get(i), list[i], "$path[$i]")
            }
            expected is JSONObject -> {
                assertNotNull(path, actual)
                for (key in expected.keys()) {
                    if (key == "schemaVersion") continue
                    val fieldName = when (key) { "centerXy" -> "centerXY"; "bottomCenterXy" -> "bottomCenterXY"; else -> key }
                    val value = if (key == "qualityScore" && actual is RiskAssessment) actual.quality.score else {
                        val field = actual!!.javaClass.getDeclaredField(fieldName); field.isAccessible = true; field.get(actual)
                    }
                    compare(expected.get(key), value, "$path.$key")
                }
            }
            actual is Enum<*> -> assertEquals(path, expected, actual.name)
            else -> assertEquals(path, expected, actual)
        }
    }
}
