package org.sightguard.app

/** Stereo gain/pan values only. This is not an HRTF or distance simulation. */
data class DirectionalAudioMix(val leftGain: Float, val rightGain: Float, val ttsPan: Float)

/** The quieter channel remains audible; callers retain the existing speech scheduler. */
data class DirectionalAudioConfig(val sideGain: Float = .2f) {
    init {
        require(sideGain.isFinite() && sideGain >= 0f && sideGain < 1f) {
            "sideGain must be finite and in [0, 1)"
        }
    }
}

object DirectionalAudio {
    /** Null is a non-directional system announcement and plays in the center. */
    fun mix(direction: Direction?, config: DirectionalAudioConfig = DirectionalAudioConfig()): DirectionalAudioMix =
        when (direction) {
            Direction.LEFT -> DirectionalAudioMix(1f, config.sideGain, config.sideGain - 1f)
            Direction.RIGHT -> DirectionalAudioMix(config.sideGain, 1f, 1f - config.sideGain)
            Direction.CENTER, null -> DirectionalAudioMix(1f, 1f, 0f)
        }
}
