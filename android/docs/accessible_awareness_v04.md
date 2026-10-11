# SightGuard Android 0.4 — Accessible Environmental Awareness

交付日期：2026-10-12（Asia/Shanghai）。本轮是现有 Android 0.3 的增量集成；没有重写风险评分，没有添加 HIGH 映射、米制测距、导航或新训练模型。

## 交付结论与检查点

- Checkpoint A：真实 SegFormer 输出接入统一环境状态机和语音。默认自动支持 WALL、STAIRS、POLE、RAILING；单级台阶/路缘真实证据不足，STEP 自动提醒关闭，保留原始 121 类观测、事件实现和音频资源。
- Checkpoint B：普通主界面精简为状态、完整预览、当前有效提醒、最近播报和四个主要操作。研究工具移到“设置 → 开发者工具”。真实 Android 回归、TalkBack、大字体、横屏、窄屏、视频文件选择及播放完成验证。
- Checkpoint C：墙面接近研究记录、轻量特征跟踪与拒绝判断机制已实现。8 组实际 AVD CameraX/模拟 IMU 场景，以及nova12当前墙面CameraX窗口已记录；不是物理手机接近正负例校准。两设备相机时钟为 UNKNOWN，拒绝将未同步 IMU 当作接近证据。
- Checkpoint D：**不启用墙面可能较近或正在接近的自动播报**。没有足够独立真机正负例证明旋转与平移可可靠区分，也没有身体运动证据。`possible_near=false`；配置加载器拒绝直接开启自动接近提示。

普通用户无需选择模型或 Mask。隐藏研究图层不关闭语义推理。未提醒不代表没有障碍。

## 仓库、基线与保存

开发目录：`D:/program/SightGuard-android-pr`，分支 `codex/android-accessible-awareness-v04`，基线提交 `456d13ffe2c5a3358637ed79e3ee624e13bfad7d`（检查时 origin/master 同提交）。已有 Wall Awareness 在该工作区内存在未提交代码；先保存，再基于它继续。原本地 HEAD、master 与源文件清单见 `../deliveries/accessible-awareness/checkpoint/baseline.json`，原 patch/源文件 ZIP 同目录保存。

主仓库 `D:/program/SightGuard` 原有分支 `codex/demo-route-geometry-v1` 和未提交修改未动。另一个产品展示 worktree 未动。首次APK交付时，本轮和继承的 Wall Awareness 修改仍为未提交本地修改；随后按用户要求在 `codex/android-accessible-awareness-v04` 提交并建立PR。未 reset、未覆盖或合并 master。首次交付的 `git-status.txt`、`git-worktrees.txt` 和源文件快照仍在交付目录，记录的是当时状态。

原 0.3 APK 保留：`../deliveries/wall-awareness/SightGuard-0.3-wall-awareness-debug.apk`，123954915 字节，SHA256 `97c7ed78a15dd669549fe865523be5030a4ffcd19c825682b927b8bb70c64795`。原 0.3 验收文档 `wall_awareness_v03.md` 保留；不得把其中旧测量当作本轮测量。

## 根因与输出链

原 master 默认关闭语义，仅在研究控件手动开启后加载 SegFormer；RiskEngine 消费 YOLO 跟踪对象，没有环境语义事件消费者。真实 ADE20K wall=0 可进入概览/研究 Grid，但没有形成正式语音请求。原本地音频缺少墙面中文句子；无离线中文 TTS 时概览仅显示文本。辅助线程原先 Depth 优先、最低约 2s 调度也会拖延反馈。原 500ms 常量与当前显示 TTL 不在同一路径，不能把所有漏报都归因于 TTL。

本轮沿用并扩展已有 Wall Awareness：

`CameraX / 文件帧 → SegFormer → 高置信度真实 Mask 连通区域 → SemanticEnvironmentEngine → SpeechController 单槽调度 → 中文 TTS / 精确本地音频 → onStart 才记为已提醒`

环境事件与 PC/Android 风险分数独立。Grid 用作研究证据，不修改风险分数；正式环境存在提示不依赖 Depth 可用。没有识别、识别但条件不足、事件提出但排队/静音/过期/被抢占、实际开始播放分别记录。

## 修改文件与职责

| 文件 | 职责 |
|---|---|
| `SemanticEnvironment.kt` | 复用并扩展纯 Kotlin 环境状态机；真实像素支持、连通区域、同一区域关联、楼梯/台阶去重、方向、连续确认、恢复、冷却和结束 |
| `EnvironmentSettings.kt`、两个 JSON 配置 | 带版本的运行参数、已验证自动类别、强制关闭未校准接近播报 |
| `SpatialAnalyzer.kt`、`SemanticEvidence.kt` | 模型 SHA、输入/输出形状、ADE20K 标签映射检查；扩展真实类别概览 |
| `SpatialSessionAnalyzer.kt`、`SpatialResourcePolicy.kt` | 单辅助工作线程；语义优先立即交付；Depth 按需加载、低频降级、无积压、会话释放 |
| `SpatialGrid.kt` | 原 Grid 保留；增加等价稀疏统计，避免为接近研究重复执行大量逐格分配 |
| `ImuMonitor.kt` | 完整帧间 IMU 窗口、时间同步质量、角速度/姿态变化；不积分加速度估算位移 |
| `WallApproachEvidence.kt`、`WallShadowAnalyzer.kt` | 特征跟踪、拟合和保守拒绝判断；有界短图像历史；同帧相对深度/Grid/地面证据 |
| `AccessibleHome.kt`、`MainActivity.kt`、Manifest | 极简可访问界面、设置/开发者入口、主动描述、诊断导出、横竖屏及原会话释放屏障 |
| `SpeechController.kt`、`res/raw/environment_*.wav` | 共用优先级队列、精确中文音频、TalkBack 音频焦点共存；保留静音、抢占、取消和停止 |
| 新增/扩展 JVM 与 instrumentation | 状态与运动拒绝测试、真实照片/模型/音频、无障碍与资源释放验收 |
| `tools/prepare_awareness_fixtures.py`、`validate_awareness_scenes.py`、`avd_camera_fixture.py` | 哈希固定输入准备、独立 PC 预检、真实 AVD 相机/音频记录；不注入检测结果 |

`core/` 原风险核心无 diff。未修改全局 Codex 配置或 AGENTS.md。

## 语义与事件参数

运行时验证 SegFormer `1×3×256×256 → 1×150×64×64` 与 ADE20K 原始 0-based ID。wall 0；stairs/stairway 53/59；step 121；pole/column 93/42；railing/bannister 38/95。聚合后仍保留原始 ID 与置信度，不推断上楼/下楼或落差方向。

先对**各原始类别**应用本类置信度阈值，再执行四邻域连通区域分析；低置信度 stairway 不能把 step 提升成已验证楼梯，也不能连接两个分离区域。支持度统计真实 Mask 单元与关注区的相交面积，而非整体 bounding box。记录面积、支持单元、面积加权置信度、P10、方向、来源与时间。连通不代表已分离出真实物体实例；相连杆柱仍可能混合。

| 事件 | 置信度下限 | ROI 覆盖 | 支持单元 | 连续观测 | 默认自动 |
|---|---:|---:|---:|---:|---|
| WALL | 0.55 | 12% | 12 | 2 | 是 |
| STAIRS | 0.55 | 3.5% | 8 | 2 | 是 |
| STEP | 0.65 | 1.5% | 5 | 3 | **否** |
| POLE | 0.60 | 1.2% | 6 | 2 | 是 |
| RAILING | 0.60 | 2.5% | 8 | 2 | 是 |

墙面关注区 x=.30–.70、y=.20–.85；杆柱/栏杆关注区扩展至 x=.08–.92，保留左右方向。不同区域独立关联；stairs/stairway/step 同一区域避免近似重复。未确认事件遇到无效/缺失观测会重置计数。确认跨度至少 250ms，源时间驱动；4s 有限恢复、15s 类别冷却、已播放稳定事件不重复。只有实际 onStart 才更新已提醒状态。确认数不是在同一帧内重复调用获得的。

源帧有效期严格 **500ms**，含推理和调度耗时；不以扩大 TTL 换成功率。3000ms 研究显示窗口不是语音有效期。快速转动、图像摘要明显变化、旧会话/跳转、明确缺失会使旧提示失效；正常源龄到期只阻止新起播，不把已经开始的一句话从中间截断。相似画面不能仅靠 RGB 摘要可靠区分，是已知边界。

Door14、Fence32、Tree4、Streetlight87、Escalator96、Water21、Rock34 为环境概览，不逐项自动播报；树冠不等于近处树干，门不等于出口，水面不等于脚下水坑。

## 真实输入与模型结果

PC CPU 预检共 300 张实际图片；20 张选入 Android CPU 真推理和声音测试，80 次推理。ADE20K 来源 `danjacobellis/scene_parse_150`，固定 revision `bbcc97e31fc582a2fe51f836cd6dccc472232aba`；其余 CC0/CC BY-SA 输入出处、许可、SHA256 在 `app/src/androidTest/assets/awareness-scenes.json`。测试图片、模型、大型输出保持 Git 忽略。下载失败及选图过程保留在预检记录；这是能力探索后选例，**不是随机抽样准确率或独立校准集**。

下表为最终 Android 输出。类别数是保留置信度≥.35 的 Mask 单元数；事件还必须满足上表更严格门限。完整所有区域/置信度/时间和声音记录见 `../deliveries/accessible-awareness/awareness-scenes-actual.json`。

| 真实场景 / 文件 | 原始目标输出 | 实际自动起播 / 失败 |
|---|---|---|
| 室内墙壁 `v03-room.jpg` | wall0=1485；有效中央覆盖30.67%，置信度.898 | 前方检测到墙面。 |
| 室内上行楼梯 `1529` | 53=32、59=415 | 本轮严格连通/支持阈值下未确认；漏报保留 |
| 室内下行楼梯 `0780` | 59=617、95=211；楼梯/扶手均有支持 | 前方检测到楼梯。不声称下行 |
| 室外楼梯 `0073` | 53=301、59=10 | 前方检测到楼梯。 |
| 夜间倾斜楼梯 `whitney-stairs.png` | stairs=0；模型高置信度误分 wall | **错误墙面播报，已知模型失败** |
| 单级门阶 `doorstep.jpg` | step121=0 | 不提示台阶，能力不足 |
| 路缘 `kerb_height_raised_original.jpg` | step121=0 | 不提示台阶，能力不足 |
| 雕刻石立柱 `1475` | column42=2684；两个独立支持区域 | 前方检测到杆状结构。 |
| 回廊立柱 `1016` | column/pole=0 | 杆柱漏检；实际背景墙触发墙面提示 |
| 路灯杆 `1662` | 87=93、93=64 | 没有满足杆柱支持阈值，不播报 |
| 阳台栏杆 `0031` | railing38=794 | 右侧检测到栏杆或扶手。 |
| 楼梯扶手 `1781` | railing38=234，扶手区域不足；59有效 | 前方检测到楼梯。扶手漏报保留 |
| 门 `0002` | door14=372 | 仅概览；实际背景墙可单独播报 |
| 园林围栏 `0041` | fence32=0 | 围栏漏检，未伪造概览 |
| 地面树影 `0027` | 53/59/121=0 | 无楼梯/台阶误报 |
| 地砖条纹 `0078` | 53/59/121=0 | 无楼梯/台阶误报；背景墙独立提示 |
| 围栏 `0411` | fence32=890 | 仅概览，无自动声音 |
| 自动扶梯 `0051` | escalator96=717 | 仅概览，无自动声音 |
| 湿地水面 `0022` | water21=866 | 仅概览，无自动声音 |
| 岩石 `0064` | rock34=1972 | 仅概览，无自动声音 |

PC 300 图中在固定≥.55 门限下未见有效 step121，Android 两个真实台阶/路缘例也失败，因此没有降低所有阈值或宣布台阶已验证。Column 成功不代表细路灯杆成功。夜间误墙面是模型泛化失败，当前没有可验证的通用规则可安全消除；不得把该输入列为成功。

重复真实静态图片可以验证模型→状态机→音频链，不等同 CameraX 移动验收。

## 中文语音与统一策略

重要风险优先，其次正常风险、环境结构、主动描述、普通状态。使用原 SpeechController 单个有期限等待槽，无各类别直接调用 TTS。普通新环境事件在后续新鲜观测可重试；已起播重要风险可抢占，过期和静音明确记录原因。繁忙/过期情况下可能延迟或丢弃提示，不能保证所有对象均被播报。

中文固定提示有精确离线 WAV；三方向 × 5 类共15种资源组合均实际 `onStart`、`completed`。全资源目录现98条，Windows Microsoft Huihui zh-CN 离线预合成；生成器及文本/哈希在本地 `validation/local_audio_manifest.json`。应用动态 TTS 只选择 `language=zh` 且不要求联网的 Voice。主动“描述周围”没有精确离线动态中文声音时显示文本及“缺少离线动态中文语音，当前描述无法朗读。”，不用通用音频冒充完整描述。

AVD最终20张照片中10个实际自动 onStart，起播源龄141–169ms，中位153ms；稳定同一事件无重复起播。完整请求、接受、抑制、取消、onStart、完成记录在 `awareness-scenes-actual.json` 和 `environment-audio-validation.json`。真机数据单独放在 `nova12/`，不与AVD合并计算性能。

实际 CameraX 环境提示亦进入本地音频：`camera-shadow-final/stationary.log` 记录 `semantic:1:0:WALL:1` proposed 后 onStart，资源 `environment_wall`；`final-session.log` 记录 `semantic:5:0:WALL:3` 实际 `elapsedMs=2276908 < validUntilMs=2276972`，中文“前方检测到墙面。”，`detail=local_media_player`。真实快速转向最终记录3次 `rapid=true` 拒绝观测，没有将这些旧证据确认成新提示。

TalkBack 开启时使用 transient_may_duck，避免强制反复夺取独占焦点；状态卡没有 liveRegion 自动连续播报。它仍受系统音频策略影响，并非所有厂商的混音体验已校准。AVD 输出 PCM 实际捕获32.63s，有非零音频；文件含 TalkBack 和应用混音，不能把全段归属于某一句，也不能替代人耳中文发音验收。

## UI 与无障碍

普通状态为“正在感知 / 正在启动 / 感知暂时受限 / 感知已停止”。完整视频预览仅 YOLO 中文标签和方框；研究 Mask 默认不显示。当前有效提醒与最近实际播报独立，未提醒原因以短中文显示。停止、静音、权限管理保持可达；静音持久保存，用户主动停止后不自行重启。

普通界面四个操作：“描述周围”“静音/恢复提醒”“停止/开始感知”“设置”。按钮至少56dp；可访问合并状态、显式焦点顺序、对比度和文字辅助，不只依赖颜色。大字体可滚动、窄屏堆叠、横屏左右布局；不依赖长按/滑块完成主要操作。研究层与原始数值只在开发者工具。

实际 TalkBack 16.0.0.738667889 服务激活，200%字体测试竖屏、横屏、420×900窄屏，四按钮可聚焦并滚动到达；3s普通自动更新未抢夺已设焦点；实际本地语音起播/完成。报告是自动化控件、焦点事件和声音验证，不声称已有视障用户完整可用性评估。测试后恢复字体、方向、分辨率和无障碍设置。

- 修改前：`../deliveries/accessible-awareness/checkpoint/ui-before.png`
- 修改后：`../deliveries/accessible-awareness/ui-after.png`
- 200%字体：`ui-font200-portrait.png`、`ui-font200-landscape.png`、`ui-font200-narrow.png`
- 文件分析：`ui-file-analysis.png`；实际从系统选择器选 `SightGuard-bus.mp4`，运行 YOLO，回到实时相机使用新会话；旧会话资源释放有日志。
- 诊断导出：实际 SAF 保存 `SightGuard-0.4-diagnostics.json`（874114字节，实际677条），含有界最多1500条元数据记录、来源/时间/原因/音频阶段，不包含相机图像。

修改前与修改后（AVD，研究控制移出普通界面）：

![修改前](../deliveries/accessible-awareness/checkpoint/ui-before.png)

![修改后](../deliveries/accessible-awareness/ui-after.png)

真机普通界面见 `../deliveries/accessible-awareness/nova12-camera-wall-override.png`；该截图时新鲜证据已过期，因此正确显示过期原因，而不会用最近一次播报冒充当前有效墙面提醒。

## 墙面接近研究与失败边界

独立状态 WALL_PRESENT / WALL_POSSIBLY_NEAR / WALL_APPROACHING / UNKNOWN / ABSTAIN；研究字段包括 wall_present、possible_near、approaching_candidate、confidence/quality、abstain_reason、frame key、源龄、连续支持、motion/depth 支持。

192像素长边灰度、同一墙面短历史，Shi–Tomasi 特征、两层 LK 和前后向检查、相似变换 RANSAC/最小二乘。至少16特征、3象限覆盖、70%内点、残差≤1.25px，旋转/横向移动/尺度异常拒绝；需连续3个扩张窗口。必须同时通过完整 IMU 区间、最大200ms传感器缺口、相机 REALTIME 同步、低角速度和姿态变化门控。特征算法有40ms工作预算；总研究步骤/系统调度并不保证每次≤40ms。

Depth 仅在同源帧、有效分布和500ms期限内统计墙面相对逆深度中位/P10/P90及同帧排名；不比较跨帧原始深度，不从逐帧归一化差值推速度。Ground/Grid 边界和覆盖记录明确空值。IMU 不积分加速度，不推手机/身体绝对位移。

即便合成输入出现可拟合扩张，也记录 `camera_translation_unverified_body_motion_unknown`，不能声称用户身体靠近墙面。低纹理、失配、旋转、多平面、遮挡、时间异常、来源变化和过期均可拒绝。窗体/手机局部移动与身体行走尚未独立验证。

AVD实际 CameraX 8组：静止、缓慢Z移动、仅左右旋转、仅俯仰、横移、远离、停止后面对墙面、无纹理天花板。73条最终研究记录，51 ABSTAIN、22 UNKNOWN，0接近候选，0接近声音；原因是IMU未同步/缺失、源过期或画面变化。AVD 真实照片是虚拟场景上的二维海报，模拟IMU不是物理真机校准；无法声称正例接近已检出。实际低纹理白墙、远墙+独立近物体、身体运动和其余受控真机正负例**尚未运行**。

首次 CameraX 实验暴露完整 Grid 统计过重，研究步骤约1.2–1.5s，导致源过期；原失败日志保留 `camera-shadow/`。已改为等价稀疏统计，并在未同步IMU时先拒绝光流；最终总研究步骤中位24ms（5–198ms），仍记录调度/GC导致的长尾。JVM用奇数641×479来源核对新旧Grid统计一致。

## 测试与性能

| 验证 | 最终结果 / 记录 |
|---|---|
| JVM | 114 passed，0 failed/errors/skipped；`final-override-build.log` |
| AVD Android instrumentation | 27项普通回归通过，215.328s；另1项TalkBack通过，26.634s；共28项，早于最终资源策略补丁 |
| 最终APK真机 instrumentation | nova12：27项全部通过，264.091s；`nova12-final-regression.log`，安装APK哈希与本地最终交付一致 |
| 实际 ONNX | YOLO、SegFormer、Depth CPU推理；标签/形状/SHA及张量对照 |
| 实际中文音频 | 15种固定环境方向资源均起播和完成；20场景自动提示10次；CameraX墙面亦有onStart |
| 真实Depth同张量对照 | 最大绝对误差3.4093857e-5；暖机244–251ms；位图推理263ms |
| 生命周期 | 3次实际启动/停止，等待主/辅助会话释放；运行PSS387739/374681/373558KB，停止183312/190461/183434KB |
| 最终停止内存 | PSS166839KB；单Activity；原会话释放日志见 final-session.log |
| CameraX全自动模式 | 8组AVD实验，YOLO中位30ms（16–70），主帧中位71ms（45–698，含启动/调度长尾），采样FPS中位13.06（0.92–20.41，含冷启动） |
| YOLO-only AVD相机 | 独立18s运行，FPS中位17.54；主帧中位48.0ms；非同步对照，不能推断真机增减幅度 |
| SegFormer实际照片 | 80次，113–159ms，中位126ms |
| CameraX语义源龄 | 中位386.5ms，228–1254ms；>500ms正确拒绝，不延长TTL |
| 静态照片语音起播源龄 | 中位153ms，141–169ms；请求到起播不是完整首次连续确认等待时间 |
| Ruff / diff | 三个修改Python工具检查/格式通过；git diff --check通过；未改PC核心，因此不重复无关PC pytest |

AVD 为 Android16/API36/x86_64，4核/1536MB。数值是该虚拟机、该实验的描述统计，不是手机性能保证。Depth并发可能使同帧深度过期，语义提示先交付不等待Depth。语义最低650ms，按耗时/主链压力退避；Depth至少6s一次并受负载门控，不强制所有模型满速。

用户此前报告红米Turbo4Pro约23FPS，未作为本轮测量。用户提供 nova12活力版/鸿蒙4.2.0，安装官方HiSuite驱动后ADB成功连接；设备报告FIN-AL60、Android12/API31、arm64-v8a、EmotionUI_14.2.0。已在此真机安装并运行模型、界面和中文声音；用户明确反馈“听到了，中文清楚”。真机回归与性能记录独立保留，不覆盖AVD记录。

### 真机发现与资源策略修复

首轮27项真机测试26通过、1失败：15种环境音频中的完成回调被另一个前台MainActivity的风险语音/音频焦点打断。原失败保留 `nova12-regression-device.log`；停止前台应用后，独立音频复测1项通过（48.925s），15组均实际onStart并完成，见 `nova12-audio-isolated.log`。这不是将失败用“生成事件”代替验收。

首轮真机20张真实照片80次SegFormer推理178–356ms、中位210.5ms；12次自动onStart，源龄250–420ms、中位304ms。此批声音存在上述前台干扰，12次起播中7完成、5取消，不能描述为全部正常完成。Depth同张量最大绝对误差3.3974648e-5，暖机1099–1275ms、位图1098ms，已超过语音源帧500ms窗口；不延长TTL，墙面存在提示不等Depth。

物理CameraX测试曾因系统thermalStatus=4/3暂停所有辅助推理。与此同时，原健康检查在每个YOLO帧重复写限制日志，导致日志洪泛；原负载退避6s又超过4s连续确认恢复窗口。这些会造成“YOLO还在运行，但墙面没有连续有效观测”的实际反馈问题。

修复后每次被拒绝的辅助请求仍消耗调度间隔；热状态2/3优先停Depth，分别保留约1.3s/2s语义间隔，主链轻度过载约3s，低内存、热状态≥4或持续严重主链过载才暂停辅助推理。Depth按需加载，不在启动时先分配模型。实际推理耗时仍可扩大间隔，不能保证低性能设备一定完成连续确认。

用户要求本次先忽略过热，因此增加**仅debug包、仅此次启动**的Intent extra `ignore_aux_thermal_for_test=true`，绕过应用辅助热门控；日志与导出记录 `thermalOverride/debugThermalOverride` 和系统真实热级别。内存、主链负载、500ms源龄和旧会话拒绝不受此开关影响；未修改Android热服务或手机系统保护，普通启动默认关闭。开关实验性能必须单独标注，不能冒充正常热降级状态。

### 真机当前墙面：真实CameraX → 中文TTS

最终APK在nova12上重新安装成功。受控启动开启上述应用测试开关，系统实际热级别4仍保留在日志。在约26s相机窗口记录32次真实SegFormer观测；第二次有效支持的中央覆盖61.66%、置信度.975、521个支持单元，事件 `semantic:1:0:WALL:1` 只提出并起播一次，原风险评分 unchanged。证据日志 `nova12-camera-wall-override.log`，截图 `nova12-camera-wall-override.png`，统计 `nova12-camera-wall-override-summary.json`。

实际起播记录：`event=onStart eventId=semantic:1:0:WALL:1 spokenText=前方检测到墙面。 elapsedMs=2519230 validUntilMs=2519379 detail=android_tts`。起播源龄351ms，请求至起播约37ms，仍在500ms期限内；用户确认“听到了，中文清楚”。随后1.628s因 `semantic_rapid_motion` 取消，**该次不记为completed**；持续同一事件没有重播。15种本地资源的完整完成验收另见独立音频测试。

此窗口SegFormer202–279ms、中位210ms，语义交付源龄274–399ms、中位285ms；采样FPS中位13.10（1.79–14.08，包含冷启动），YOLO中位53ms、主帧74ms。Depth990–1084ms、中位997ms，导致同帧研究结果过期时正确拒绝；不会阻断先交付的墙面提示。辅助总研究步骤25–116ms、中位34.5ms；进程PSS快照424932–502359KB、中位484701KB，截图时dumpsys总PSS474881KB。这些是应用热门控绕过实验，不能当作所有手机或正式默认策略的FPS保证。

真实CameraX时钟仍报告UNKNOWN。32条研究记录中27 ABSTAIN、5 UNKNOWN，26次IMU不同步/缺失，5次源过期、1次帧间时间异常；0 possible_near、0 approaching_candidate。设备IMU采集和快速转动门控确实运行，但没有将未校准时钟强行视为同步。因此没有声称完成物理接近正负例校准，也没有启用接近自动播报。

### 最终APK真机完整回归

最终包27项通过、0失败（264.091s），见 `nova12-final-regression.log`。本轮在开始前关闭旧前台应用，避免首轮音频焦点干扰；测试仍实际使用CameraX、ONNX和播放器，没有跳过或弱化断言。最终结果与Mask图片单独保留 `nova12-final/`，首轮失败与独立复测记录均保留。

- 20张真实照片、80次推理：SegFormer175–325ms，中位187.5ms。10次自动环境提示均实际onStart并completed，起播源龄217–381ms、中位228.5ms；另4次请求被抑制并记录原因，未冒充播放成功。模型漏检/误墙面结论与上表一致。
- 15种固定环境方向资源：15次requested、accepted、onStart、completed；0取消。真实墙面静态图状态机测试确认稳定去重、离开后结束、过期请求取消与原风险语音隔离。
- Depth暖机892–900ms、中位893ms，位图903ms，同张量最大误差3.3974648e-5。独立Depth测试PSS173760→314851→311431KB，close后瞬时PSS未恢复到起点；原生分配器保留和采样时机仍影响内存，不能声称close即回收全部RSS/PSS。
- 3次真实相机启动/停止：运行PSS322792/344179/311479KB，等待模型关闭后的PSS311276/308127/296183KB；主/辅助释放完成标志和modelReady=false均验证。该短测未见持续递增，但不等同长时间无泄漏保证。
- TalkBack服务、200%字体、横竖屏/窄屏专项是在AVD完成；**未在nova12上完成TalkBack专项**，不把普通无障碍控件树测试等同厂商屏幕阅读器的人机体验验收。

结束后应用已force-stop，清除本次启动的热测试开关并释放相机/模型；偏好确认为auto_start=true、muted=false。下一次普通启动使用默认资源策略。AVD已关闭，没有持续占用qemu虚拟机内存。

保留开发失败：第一次光流纹理夹具没有足够Shi–Tomasi角点，分数像素锐利棋盘重采样被固定光度门限拒绝。改用已知无损整数平移验证基本LK路径，未降低生产门限；`pixel-flow-test.log`保留失败，`pixel-flow-test-integer.log`为最终通过。合成扩张测试通过不等同真实接近检测成功。

## 安装包与复现

最终 APK：`D:/program/SightGuard-android-pr/android/deliveries/accessible-awareness/SightGuard-0.4-accessible-awareness-debug.apk`

225398925字节（包含原先未打包的Depth模型），SHA256：

`6671a81d740d5a12087f7af6eb8796b7565f8c28add75f59ffbcc5f022753fd7`

versionCode4 / versionName0.4-accessible-awareness，minSdk26，arm64-v8a/armeabi-v7a/x86_64。最终补丁已在nova12真机安装运行并通过27项回归；此前版本在AVD通过28项。手机已安装base.apk的SHA256与本地最终APK完全一致。APK内模型：

- YOLO26n：`e093d7e1805d63cc1b671c333ce0b48b5cd33f70f64ea16c9b5bd713e33242cf`
- SegFormer：`85913f7adae1b787da8cbe4a046e6d8a22bae901340d548be2fea46ec7ff2e0f`
- Depth Anything V2：`01013aabdd38909eff174b88c177ec583967b37d14041bab8f91d405c3a1760e`

```powershell
. ./android/tools/task_environment.ps1
# 大型模型/真实图片按既有模型说明和固定SHA准备；默认JVM测试不下载、不依赖GPU。
./android/gradlew.bat -p android testDebugUnitTest assembleDebug assembleDebugAndroidTest --console=plain
& D:/program/SightGuard/.venv/Scripts/python.exe -X utf8 android/tools/prepare_awareness_fixtures.py
# 首次缺少真实图片时显式追加 --download；默认不会联网下载。
adb -s emulator-5554 install -r android/app/build/outputs/apk/debug/app-debug.apk
adb -s emulator-5554 install -r android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb -s emulator-5554 shell am instrument -w -r -e notClass org.sightguard.app.TalkBackLayoutInstrumentedTest org.sightguard.app.test/androidx.test.runner.AndroidJUnitRunner
# TalkBack测试需先安装并启用实际服务，单独执行，结束后恢复测试前设置。
adb -s emulator-5554 shell am instrument -w -r -e class org.sightguard.app.TalkBackLayoutInstrumentedTest org.sightguard.app.test/androidx.test.runner.AndroidJUnitRunner
```

原数据集、图片选择及下载失败、本轮输入/模型哈希、机器JSON、日志、截图均保留在本地 `deliveries/accessible-awareness/`。测试结果不是安全保证；存在真实漏检和误识别，环境辅助继续补充盲杖与导盲犬。
