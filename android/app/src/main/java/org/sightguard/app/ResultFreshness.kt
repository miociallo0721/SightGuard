package org.sightguard.app

/** UI/feedback freshness uses local monotonic receive time, separate from camera event time. */
object ResultFreshness {
    const val MAX_AGE_MS = 1500L
    fun isCurrent(receivedAtMs: Long, nowMs: Long): Boolean =
        receivedAtMs >= 0 && nowMs >= receivedAtMs && nowMs - receivedAtMs <= MAX_AGE_MS
}
