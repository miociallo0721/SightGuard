# Annotation Studio 验证记录

日期：2026-10-11（Asia/Shanghai）。本次只验证工具，不由开发代理完成真实风险真值。

## 自动化与构建

新增 `tests/test_annotation_studio.py` 共 16 项（含参数化用例），使用本地生成的
64×48、10 FPS、20 帧 MP4 与 Mock 感知记录，不下载模型或依赖 GPU。
覆盖：注册记录/原始路径与身份、逐帧/PTS/VFR 选择、时间轴新增修改删除、越界、
非法框与机会、同类重叠、Draft 保存重载及 SHA 绑定、未保存/并发防护、
空 events 与人工负例确认、ready/reviewed 状态及第二审核者、Benchmark 草稿拒绝、
历史证据、写入失败回滚、视觉属性隔离、无模型输出的 Viewer、路径逃逸、
以及 Gradio 键盘编辑后数字字符串转换。

真实 UrbanEgo 八份原始标签没有在自动化测试中标为 reviewed。审核迁移测试的
“real”输入是测试目录中的人工合成视频，用于验证代码入口，不是道路测试真值。

运行检查：

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest -q --basetemp models/hybrid/tmp/pytest-studio-release
uv build --out-dir outputs/console/annotation-studio-final-build
```

Ruff 与格式检查通过。pytest **348 passed，77.61 秒，无跳过**；运行时打开
`SIGHTGUARD_TEST_MODEL`（本地 YOLO 权重）、`SIGHTGUARD_TEST_HYBRID=1`、
`SIGHTGUARD_TEST_HYBRID_GPU=1`，包含原有真实模型回归。
最后补充的存储防护及页面构建检查 **17 passed，5.73 秒**，包含 16 项 Studio
测试及原有本机无模型加载 UI 启动测试；最终 UI 按钮还经过浏览器检查，详见下节。

构建成功生成 sdist 和 wheel；解包检查 wheel 含两个 Studio 模块，初次 sdist 471,941
字节、197 项，无模型权重、MP4 或 Hybrid 缓存。uv 提示缓存位于源目录内，本次
实际归档检查确认没有收录这些大文件。项目版本仍为 0.4.0，未创建发布标签。

最终重新构建产物位于 `outputs/console/annotation-studio-final-build/`；sdist 199 项，
确认不包含模型、视频、PNG 原图/截图、Hybrid 缓存或浏览器测试数据。
wheel 内含最终 `fully_excluded` 存储处理和表格数值转换代码。

首次全量回归发现 Studio 初始选择没有原 MP4 的旧合成 manifest 导致页面构建失败，
已改为过滤没有原视频的数据集，保留其他页面原有行为。针对性检查通过，完整回归
重新通过。没有删改或弱化原有效测试。

## 真实 Chrome 操作

启动 `uv run sightguard ui --research --port 7861`。使用 Playwright CLI 0.1.22
驱动本机 Chrome，窗口 **1440×1000**，实际点击/键盘操作页面，不以 HTTP 200
代替交互验收。浏览器会话、npm 缓存、Gradio 缓存与产物均配置在 D 盘。

隔离 manifest：`datasets/annotation_studio_validation/manifest.csv`，引用真实
`urbanego_run8_01` 的原 MP4/感知数据，但标签复制到独立 annotations 目录。
写入作者为 `UI_TEST_ONLY`，文件备注明确不是人工 Ground Truth；始终为 draft。

已实际验证：

1. 打开 Annotation Studio、选择隔离副本并加载真实原画面，无模型框/分数/提醒。
2. 下一帧由 0 ms 移到 50 ms；指定 7250 ms 对应原帧定位，播放与暂停。
3. 创建测试事件及 initial 机会，保存 Draft。
4. 编辑结束时间后尝试加载，被 Unsaved edits 提示阻止。
5. 应用/保存 0–1500 ms 区间后，重新加载、点击时间轴选择事件。
6. 将区间改为 100–1700 ms，机会表开始点改为 100 ms；保存、重新加载后
   开始与结束值完全恢复，未取整成秒。
7. Schema 验证显示有效；Benchmark 按原校验器拒绝 draft，没有监督指标。
8. 删除测试事件并保存；进度显示一条 draft，不是 reviewed 负例。
9. 加载 7500 ms 原始关键帧并保存独立视觉属性测试文件，风险标签未改。
10. 最终 UI 验证播放后暂停保持、无法判断区间创建/保存/验证/删除、添加升级机会
    与人工目标区域行，以及明确丢弃未应用表单后重新加载。

表格第一次键盘编辑返回数字字符串，被 Schema 正确拒绝；已修正 UI 的数值转换，
增加回归测试并重新成功跑通，未放宽 Schema。Timer 的迟到响应不再重新启用暂停播放。

截图与操作脚本仅保存在本地 `outputs/console/annotation-studio-*.png` 和
`outputs/console/studio-*-browser-flow.js`，不提交截图或真实原图。原始八份标签与
manifest 保存了测试前后 SHA256 对照，确认未污染；人工完成数量仍为 **0**。

## 本轮未验证或不包含

没有实际人类完成标注或二次复核；审核身份、独立观看由使用者声明，程序不能认证。
没有真实 A/B/C 监督指标或 Laya Vision 准确率。没有多浏览器兼容性、压力测试、
长时间播放稳定性或断电恢复验收。可捕获文件替换失败已模拟回滚，突然断电时多个
文件之间的完整事务原子性不能保证。

本版没有拖拽框/区间、自动保存、关闭页面恢复或多用户权限系统；这些不是现有
人工闭环的前置条件。目标区域通过数值填写，时间轴通过选择与数值/当前帧设置编辑。
