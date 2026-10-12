# 版本与交付

当前 Android 源码版本为 **0.8.1**，Python 包版本为 **0.4.0**。两者独立版本化。

APK 由本机交付目录提供，不随 Git 提交；当前仓库 Releases 没有可下载的 APK 资产。因此本页不提供无效的“立即下载”链接。获取交付包后可用以下哈希核对，或按 [Android 构建指南](development/android.md)自行构建。

| Android 版本 | 本地交付文件 | 说明 |
| --- | --- | --- |
| 0.8.1 | `android/deliveries/local-space/SightGuard-0.8.1-preview-band.apk` | 新增实验候选带；真机阳性绘制验收待补 |
| 0.8 | `android/deliveries/local-space/SightGuard-0.8-local-space.apk` | 局部空间描述基线 |
| 0.7 | `android/deliveries/yoloe26-camerax/SightGuard-0.7-YOLOE-CameraX.apk` | YOLOE / CameraX 基线 |

SHA256：

```text
0.8.1  2e842ee049256e65a5dab8eaeb726f0e28fd0293bf7d5022ff6d3cc4ef8a53fb
0.8    bab43c6c4fbb119a769942b7d8e7b34ed09c4ade4568b58847d4c2d4d186712f
0.7    5315f7c4aba7c344c341e11aa7976674ae1ff254bc94d06f7ec4bb95e6a01310
```

这些哈希只对应已有交付文件，不对应任意重新构建的 APK。用户可见变化见 [CHANGELOG](../CHANGELOG.md)，回退要求见[发布流程](development/releasing.md)。
