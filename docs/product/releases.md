# 版本、交付与回退

当前没有正式签名的公开发行版本。以下调试包和验证记录保留在本机 Git 忽略的 `android/deliveries/`，不是 GitHub 中已发布的附件。克隆仓库不会自动取得它们。

## 稳定回退基线

- 版本：`0.4-accessible-awareness`，versionCode 4。
- 源码：`4a1fdfb97190a9a85e151479ec19a13b8c37bc73`，对应 0.4 成果，先前 PR #6 已合并。
- 本机包：`android/deliveries/metric-corridor-research/baseline/SightGuard-0.4-stable.apk`。
- SHA256：`6671a81d740d5a12087f7af6eb8796b7565f8c28add75f59ffbcc5f022753fd7`。
- 约 225MB。既往 nova12 真机最终回归 27 项通过；TalkBack 专项另在 AVD 验证。详见[0.4 验收](../../android/docs/accessible_awareness_v04.md)。

## 米制 Shadow 实验包

- 版本：`0.4-metric-corridor-shadow`，versionCode 5。
- 源码：`b3d0aeb009356dcd8625d9f48f3d2138434b7623`，[PR #7](https://github.com/miociallo0721/SightGuard/pull/7)。PR 状态以 GitHub 当前显示为准。
- 本机包：`android/deliveries/metric-corridor-research/SightGuard-metric-corridor-experimental.apk`。
- SHA256：`72ad507807ffb54b570a11996da38cc4c588ce5b78f83376575fcf316ce88d0a`。
- 约 325MB。新增模块默认关闭，不接入风险与语音。JVM 131 项通过，PC 同版本 ONNX Runtime 数值对照通过；不等同 Android 新模型推理验证。
- 后续安装记录：`android/deliveries/metric-corridor-install-20261012/installation-verification.json`。2026-10-12 09:23 北京时间 ADB 安装成功、版本和 MainActivity 前台已核对，之后设备断开。没有新米制 CameraX / 性能验收。

旧 `metric-corridor-research/final_report.md` 和 `rollback.md` 仍保留当时安装未完成的状态；当前安装事实以上述独立后续记录为准，不能据此修改旧日志。

## 回退方式

首先关闭“设置 → 开发者工具”的米制研究开关。新增研究不应改变正式语音；若需要完整回退，使用已保存稳定包，在已正常授权的设备上执行：

```powershell
# 从仓库根目录执行；将 AUTHORIZED_SERIAL 替换为已授权设备序列号
$adb = "$env:ANDROID_HOME/platform-tools/adb.exe"
& $adb -s AUTHORIZED_SERIAL install -r -d -t android/deliveries/metric-corridor-research/baseline/SightGuard-0.4-stable.apk
```

两包使用同一调试证书；`-d` 是调试包降版本选项，不保证所有厂商接受降级。若系统拒绝或要求人工审批，应保留当前应用，不卸载、不清空数据、不绕过权限。用户数据兼容性仍需按版本验证。

源码可从基线 SHA 在新的空目录创建独立 worktree；不要在用户 dirty checkout 上 reset，不删除其他 worktree。

## 发布前要求

发布负责人应对下面的实际结果签字确认；清单的存在不代表项目已经满足要求。

- 确认整体 LICENSE、权重再分发、第三方 NOTICE、数据署名和音频资源来源；未明确的许可不能推断为允许发行。
- 在干净克隆中准备经过哈希核验的模型，完成构建、核心测试和 APK 资产 / 权限 / 签名检查，记录 Git SHA 与锁文件版本。
- 使用目标设备完成 CameraX、真实模型、源龄、快速转向、静音、停止、后台、文件切换和实际中文 `onStart` 验收。
- 完成 TalkBack、字体放大、权限拒绝和感知受限流程；新增研究缺乏独立部署验证时保持关闭。
- 在独立来源数据上评价重要风险召回、无效提醒、重复和延迟，保留失败与弃权，不把公开数据评测等同视障用户道路验证。
- 完成长会话内存、热状态、续航及低内存回归；给出实际支持设备和降级策略。
- 使用受控发行签名，不提交私钥、密码或 keystore。制定升级 / 回退兼容性和问题报告渠道。
- 为每个交付包保存独立名称、SHA256、源码版本、模型契约、配置、测试报告、失败记录和回退说明，不覆盖旧包。

本轮公开数据集任务没有可用手机；如只发生 PC / 文档修改，不为了更新 README 重复打包或宣称新增 Android 功能。
