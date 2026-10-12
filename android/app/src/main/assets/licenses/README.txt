SightGuard research build — model provenance

SegFormer MiT-B0 ADE20K uses the frozen OpenMMLab checkpoint:
https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth
SHA256: 8ffa8fdac140ffaef9119150aca65e9fcb4e7a165e60b80be0102cc805d73a9c

Architecture: SegFormer, NVIDIA (https://github.com/NVlabs/SegFormer).
Original repository license: SegFormer-NVIDIA.txt, includes research/evaluation-only
non-commercial use limitations. This build is delivered for research/evaluation.
MMSegmentation framework license: MMSegmentation-Apache-2.0.txt.
The framework license does not by itself establish unrestricted rights for every
checkpoint or upstream work. Commercial distribution requires a separate rights review.

0.6 changes export the same weights to a 384 input in addition to the original 256
input, preserve preprocessing and validate PyTorch/ONNX/Android numerical agreement.
No ADE20K evaluation photographs or annotations are included in the application.
Other models retain their existing provenance in the model manifests and project docs.
