package org.sightguard.app
import org.junit.Assert.*
import org.junit.Test
class AutoStartPolicyTest {
 @Test fun permissionIsRequestedOnceAndDenialDoesNotLoop() { val p=AutoStartPolicy(); assertEquals("PERMISSION",p.enter(false)); assertNull(p.enter(false)); p.leave(false); assertNull(p.enter(false)) }
 @Test fun coldLaunchAndPermissionGrant() { val p=AutoStartPolicy(); assertEquals("START",p.enter(true)); assertNull(p.enter(true)); assertTrue(p.permissionGranted(false)); p.leave(false); assertFalse(p.permissionGranted(false)) }
 @Test fun explicitStopPreventsResumeAndRecomposition() { val p=AutoStartPolicy(); p.enter(true); p.manualStop(); p.leave(false); assertNull(p.enter(true)); assertNull(p.enter(true)); p.manualStart(); assertTrue(p.permissionGranted(true)) }
 @Test fun backgroundResumesOnlyRunningEnabledCamera() { val p=AutoStartPolicy(); p.enter(true); p.leave(true); assertEquals("START",p.enter(true)); p.enabled=false; p.leave(true); assertNull(p.enter(true)) }
 @Test fun fileModeAndFailureDoNotRestart() { val p=AutoStartPolicy(); p.enter(true); p.sourceChanged(true); p.leave(true); assertNull(p.enter(true)); p.sourceChanged(false); p.failed(); p.leave(false); assertNull(p.enter(true)) }
 @Test fun pendingReleaseRetainsResumeIntentAcrossSecondBackground() { val p=AutoStartPolicy(); p.enter(true); p.leave(true); assertEquals("START",p.enter(true)); p.leave(false,true); assertEquals("START",p.enter(true)); p.manualStop(); p.leave(false,true); assertNull(p.enter(true)) }
 @Test fun disabledColdLaunchAndSavedStop() { assertNull(AutoStartPolicy(false).enter(true)); assertNull(AutoStartPolicy(true,true).enter(true)) }
}
