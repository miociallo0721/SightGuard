package org.sightguard.app

/** Session-local, one-way fallback. Never extends evidence validity or stacks models. */
class SemanticResolutionPolicy(
    private val maxAgeMs: Long = 400,
    private val consecutive: Int = 2,
    private val primarySlowLimit: Int = 5,
) {
    init { require(maxAgeMs>0 && consecutive>=2 && primarySlowLimit>0) }
    private var slowObservations = 0
    var fallback = false; private set
    fun observe(sourceAgeMs: Long, primarySlowFrames: Int, failed: Boolean = false): Boolean {
        slowObservations = if (sourceAgeMs > maxAgeMs || sourceAgeMs < 0) (slowObservations+1).coerceAtMost(consecutive) else 0
        if (failed || slowObservations >= consecutive || primarySlowFrames >= primarySlowLimit) fallback = true
        return fallback
    }
}
