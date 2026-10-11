package org.sightguard.app

import org.junit.Assert.*
import org.junit.Test

class MetricResearchTest {
    private val key=SpatialFrameKey(1,0,1,100.0,640,480,1000)
    private val shape=MetricShape(64,64)
    private fun semantics(classId: Int=0,confidence: Float=.9f)=GroundEvidence(key,
        listOf(SemanticCell(0f,0f,1f,1f,classId,confidence)),20,1020,"synthetic_unit_fixture")
    private fun depth(value: Float=2f)=MetricDepthEvidence(key,FloatArray(4096) {value},shape,40,1040,"synthetic_unit_fixture")
    private fun estimate(s: GroundEvidence?=semantics(),d: MetricDepthEvidence?=depth(),
                         now: Long=1050,scene: Boolean=true,boxes: List<Detection> = emptyList())=
        WallMetricEstimator().estimate(key,s,d,boxes,now,scene)

    @Test fun officialAspectResizeDoesNotWarpUnsupportedInput() {
        assertEquals(MetricShape(280,378),MetricDepthMath.shape(640,480))
        assertEquals(MetricShape(378,280),MetricDepthMath.shape(480,640))
        assertEquals(MetricShape(504,280),MetricDepthMath.shape(720,1280))
        assertEquals(MetricShape(280,504),MetricDepthMath.shape(1280,720))
        assertNull(MetricDepthMath.shape(600,600))
    }

    @Test fun cubicConstantRgbRetainsNormalizationWithoutPadding() {
        val source=IntArray(16*12) {0xffff0000.toInt()}
        val target=MetricShape(280,378)
        val values=MetricDepthMath.prepare(source,16,12,target)
        val plane=target.height*target.width
        assertEquals(((1-.485)/.229).toFloat(),values[0],.00001f)
        assertEquals((-.456/.224).toFloat(),values[plane],.00001f)
        assertEquals((-.406/.225).toFloat(),values[plane*2],.00001f)
        assertEquals(values[0],values[plane-1],.00001f)
    }

    @Test fun cubicDownsamplingMatchesIndependentOpenCvBoundaryAndOvershootProbes() {
        // Level 1: independent OpenCV float64 INTER_CUBIC reference, not phone validation.
        val pixels=IntArray(640*480) {i->val x=i%640;val y=i/640
            (255 shl 24) or (((x*17+y*3)%256) shl 16) or (((x*5+y*11)%256) shl 8) or (((x/16+y/16)%2)*255)}
        val shape=MetricShape(280,378);val tensor=MetricDepthMath.prepare(pixels,640,480,shape)
        val probes=listOf(0 to 0,1 to 1,377 to 0,0 to 279,377 to 279,17 to 29,
            188 to 139,189 to 140,277 to 219,55 to 99,5 to 101,336 to 51)
        val expected=arrayOf(
            floatArrayOf(-2.0248603f,-1.9582751f,-1.8044444f),floatArrayOf(-1.4111244f,-1.4512512f,-1.8044444f),
            floatArrayOf(-.2814073f,.1477604f,2.64f),floatArrayOf(.6350337f,.542715f,2.64f),
            floatArrayOf(-.5623283f,-2.2147777f,-1.8044444f),floatArrayOf(.1804716f,1.2023021f,-1.8044444f),
            floatArrayOf(2.5530327f,.0967098f,3.734439f),floatArrayOf(-2.0336189f,.5937663f,3.7344367f),
            floatArrayOf(.3929467f,-.635182f,-1.8044444f),floatArrayOf(-1.2377901f,-1.4320876f,2.64f),
            floatArrayOf(.5791988f,.7710564f,-1.8044444f),floatArrayOf(1.5301044f,1.9476395f,-1.8044444f))
        probes.forEachIndexed {i,(x,y)->for(channel in 0..2)
            assertEquals("OpenCV probe=$i channel=$channel",expected[i][channel],tensor[channel*shape.width*shape.height+y*shape.width+x],.0003f)}
    }

    @Test fun constantMetricWallIsPredictionOnly() {
        val record=estimate()
        assertEquals("PREDICTED_METRIC_DEPTH",record.status)
        assertEquals(2f,record.predictedMetricDepth!!,0f)
        assertEquals(1f,record.validPixelFraction,0f)
        assertTrue(record.wallPresent==true)
        assertFalse(record.automaticAlertEnabled)
    }

    @Test fun staleEvidenceDoesNotCreateCurrentDistance() {
        val record=estimate(now=1501)
        assertEquals("ABSTAIN",record.status);assertNull(record.predictedMetricDepth)
        assertTrue(record.rejectionReasons.contains("source_stale"))
    }

    @Test fun sceneChangeAndSourceMismatchReject() {
        assertTrue(estimate(scene=false).rejectionReasons.contains("source_scene_changed"))
        assertTrue(estimate(d=depth().copy(key=key.copy(frameIndex=2))).rejectionReasons.contains("depth_frame_mismatch"))
        assertTrue(estimate(s=semantics().copy(key=key.copy(generation=2))).rejectionReasons.contains("semantic_frame_missing_or_mismatch"))
    }

    @Test fun relativeUnitsAndFutureCompletionAreRejected() {
        assertTrue(estimate(d=depth().copy(quantity="relative_inverse_depth")).rejectionReasons.contains("depth_is_not_metric_z"))
        assertTrue(estimate(d=depth().copy(completedAtMs=999999)).rejectionReasons.contains("depth_completion_clock_invalid"))
    }

    @Test fun wallPresenceSurvivesUnavailableDepth() {
        val record=estimate(d=null)
        assertTrue(record.wallPresent==true);assertNull(record.predictedMetricDepth)
        assertTrue(record.rejectionReasons.contains("metric_depth_unavailable"))
    }

    @Test fun missingWeakAndPeripheralSemanticsDoNotInventWall() {
        assertNull(estimate(s=null).wallPresent)
        assertTrue(estimate(s=semantics(3)).wallPresent==false)
        assertTrue(estimate(s=semantics(confidence=.2f)).wallPresent==false)
        val edge=semantics().copy(cells=listOf(SemanticCell(0f,0f,.1f,1f,0,.9f)))
        assertTrue(estimate(s=edge).wallPresent==false)
    }

    @Test fun invalidAndSaturatedMetricPixelsAreExcluded() {
        for(value in listOf(Float.NaN,Float.POSITIVE_INFINITY,0f,-1f,20f)) {
            val record=estimate(d=depth(value))
            assertEquals(0f,record.validPixelFraction,0f);assertNull(record.predictedMetricDepth)
        }
    }

    @Test fun foregroundProxyExcludesWallPixels() {
        val box=Detection(0,.99f,0f,0f,1f,1f)
        val record=estimate(boxes=listOf(box))
        assertEquals(0,record.wallPixels);assertFalse(record.automaticAlertEnabled)
    }

    @Test fun rapidDeviceMotionRejectsWithoutClaimingImuSync() {
        val record=WallMetricEstimator().estimate(key,semantics(),depth(),emptyList(),1050,
            currentMotion=MotionEvidence(MotionQuality.RAPID,"synthetic gyro gate"))
        assertNull(record.predictedMetricDepth)
        assertTrue(record.rejectionReasons.contains("current_device_rapid_motion"))
        assertEquals("MOTION_SYNC_UNKNOWN",record.motionSync)
    }

    @Test fun missingModelIdentityCannotBecomeMetricEvidence() {
        assertNull(estimate(d=depth().copy(modelSha="")).predictedMetricDepth)
    }

    @Test(expected=IllegalArgumentException::class) fun invalidFractionConfigFailsClosed() {
        WallMetricConfig(minimumConfidence=Float.NaN)
    }

    @Test fun realtimeCaptureAgeRejectsAnOldFrameEvenWhenReceiptIsRecent() {
        val actual=key.copy(timestampClock="REALTIME",captureTimestampNs=500_000_000L)
        val record=WallMetricEstimator().estimate(actual,semantics().copy(key=actual),depth().copy(key=actual),emptyList(),1050)
        assertEquals(550L,record.sourceAgeMs);assertNull(record.predictedMetricDepth)
        assertEquals("REALTIME_sensor_capture",record.sourceAgeBasis)
        assertEquals(1050L,record.generatedAtElapsedMs)
    }

    @Test fun unknownClockDoesNotInventAnOffsetOrPreciseExposureAge() {
        val actual=key.copy(captureTimestampNs=999_999_999_999_999L)
        val record=WallMetricEstimator().estimate(actual,semantics().copy(key=actual),depth().copy(key=actual),emptyList(),1050)
        assertEquals(50L,record.sourceAgeMs)
        assertEquals("analyzer_receipt_or_decode_request_not_exposure_age",record.sourceAgeBasis)
        assertEquals("MOTION_SYNC_UNKNOWN",record.motionSync)
    }

    @Test fun futureRealtimeCaptureIsRejected() {
        val actual=key.copy(timestampClock="REALTIME",captureTimestampNs=1_100_000_000L)
        val record=WallMetricEstimator().estimate(actual,semantics().copy(key=actual),depth().copy(key=actual),emptyList(),1050)
        assertNull(record.predictedMetricDepth);assertTrue(record.rejectionReasons.contains("source_stale"))
    }
}
