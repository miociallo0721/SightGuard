package org.sightguard.app

/** Same frozen checkpoint, separately exported static shapes; never reinterpret an asset. */
object SegFormerModels {
    data class Contract(val inputSize: Int,val file: String,val sha256: String) {
        val outputSize: Int get()=inputSize/4
    }
    const val DETAIL_SHA="49166600bda475ede43b413e8b145b735fe4532361fa663646e237f19e7dad69"
    const val BALANCED_SHA="81ef9abfd868c0fe3ab62859145995dfc89212cc1ff5da311957c0aea0152fc8"
    fun contract(size: Int): Contract = when(size) {
        256 -> Contract(256,SpatialDetector.MODEL_FILE,SpatialDetector.MODEL_SHA)
        320 -> Contract(320,"segformer_mit_b0_ade20k_320.onnx",BALANCED_SHA)
        384 -> Contract(384,"segformer_mit_b0_ade20k_384.onnx",DETAIL_SHA)
        else -> error("Unsupported SegFormer contract: $size")
    }
}
