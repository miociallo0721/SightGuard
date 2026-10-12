> 历史资料：保留当时的实现与验证记录。当前使用入口见仓库 README；本文不是当前能力清单。

# Spatial 开发暂停与续作记录

2026-10-10 曾按用户要求暂停，随后用户明确要求继续；已恢复并完成收尾。禁止自动推送仍有效。
工作目录 D:/program/SightGuard，master 基线 757468e；当前代码和文档均保留为未提交修改。
AGENTS.md 未改，risk/decision 原有评分和状态机未改，没有 push。

## 当前完成

- spatial/ 提供 ctypes OpenAL Soft HRTF 后端、配置、可序列化 cue、坐标转换、短音生成、
  有界优先级/TTL/去重/抢占调度、线程 worker、RiskCueAdapter、NavigationAdapter。
- CLI spatial-demo 与 opt-in live --spatial-audio，默认静音功能未改变视觉规则。
- 研究控制台加入 Spatial Audio Lab；原产品网页7860直接加入实时摄像头、
  音频开关、实际HRTF设备摘要、左前右试听、导航模拟控制。上传与回放保留。
- 用户已确认耳机可听见、左右正确；前后与高度未验收。
- 用户发现 Xiaomi 画面黑色彩条。原始 DirectShow 默认 YUY2 同样损坏。
  设置尺寸/FPS后最后设置MJPG修复；配置camera_fps30/camera_fourccMJPG。
  5秒实机CLI获得正常画面、检出行人、既有first与perception_limited告警；用户确认正常。
- 原产品网页已经重启加载最新代码，实机网页显示正常图像、事件与提醒、真实HRTF。
  暂停时点击停止；网页确认“已停止 · 摄像头与推理线程已释放”，开始按钮可用。

## 已完成验证

- uv run ruff check src scripts tests：All checks passed。
- uv run ruff format --check src scripts tests：95 files already formatted。
- uv run pytest -q：282 passed, 2 skipped（默认无本地模型集成）。
- SIGHTGUARD_TEST_MODEL=models/yolo26n.pt；uv run pytest tests/test_integration.py -q：2 passed。
- uv build：0.4.0 wheel/sdist成功，wheel包含spatial和网页资源。
- node --check src/sightguard/ui/web/app.js：通过（前端代理执行）。
- 合成检测经真实AssessmentEngine产生risk_initial/risk_escalation，真实OpenAL均提交播放并正常关闭。
- OpenAL Soft 1.25.2，Realtek输出，HRTF enabled / Built-In HRTF。

## 恢复时优先事项

1. 检查git diff和工作目录；不覆盖现有修改。不要改AGENTS.md。
2. 补采集配置顺序回归测试：WIDTH HEIGHT FPS FOURCC(MJPG)最后，BUFFERSIZE随后。
   spatial_tests代理刚开始该测试即被暂停，未写入；原3项live-spatial测试已完成并通过。
3. 收尾产品页实机验收：静音、导航暂停/恢复/取消、页面隐藏/退出释放；
   默认checkbox off，不能硬编码风险，不要把旧帧结果贴到新帧。
4. 更新docs/spatial_audio_validation.md的最终完整测试汇总（当前正文尚未写282passed），
   记录网页实际验证；docs/architecture.md可补一句spatial职责；完成必要检查后停止扩展。
5. 可创建本地feature提交（此前用户允许），不得自动push。尚未提交当前改动。
   Git身份已有Mio / miociallo0721@gmail.com，不要更改全局配置。
6. 原生DLL/权重/私人视频/输出仍Git忽略。不要将camera_diagnostic照片提交。
   scripts/prepare_openal.py已写，已有DLL保留分支验证；脚本下载分支未重新运行。

## 服务与文件

- 产品网页 http://127.0.0.1:7860/，PID26972（exec session56942），保留运行，无摄像头会话。
- 研究台7861 PID28536（session96689）和独立Lab7862 PID32016（session28950）仍在；
  它们启动早于摄像头格式修复，若使用应重新启动。没有播放会话。
- 音频独占只在同进程；不要同时在多个服务进程播放。
- 文档：docs/spatial_audio_architecture.md、docs/spatial_audio_validation.md。
- 忽略的本地记录：outputs/spatial_validation/p0_headphones.json（用户确认已更新）、
  p1_synthetic_native.json；outputs/camera_diagnostic/保存故障诊断原始帧。
- 真实听感不等于道路效果或安全保证；尚未完整性能/长期稳定性/前后高度/头部跟踪验证。


## 恢复后状态

采集配置顺序测试已补齐，完整回归284passed/2skipped，Ruff/格式/JS语法通过。
产品页静音、导航控制、停止与实际HRTF状态已检查；文档汇总已更新。
不再扩展开发；尚未专项验证真实道路效果、长时稳定性、前后高度或隐藏事件切页。
