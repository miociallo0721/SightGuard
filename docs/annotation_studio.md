# Annotation Studio：独立人工 Ground Truth

Research Console 新增 Annotation Studio 页，用于原视频逐帧检查、人工时间轴编辑、
Schema 3.0 草稿保存及二次审核。Risk Engine、YOLO/BoT-SORT、AlertEvent、音频和
Phase 3 评价协议保持不变；不运行模型辅助标注，不下载数据。

## 启动

从 Windows 项目根目录执行，缓存及临时文件使用 D 盘：

```powershell
Set-Location D:\program\SightGuard
$env:TEMP = "$PWD/models/hybrid/tmp"
$env:TMP = $env:TEMP
$env:TMPDIR = $env:TEMP
$env:UV_CACHE_DIR = "$PWD/models/hybrid/cache/uv"
$env:HF_HOME = "$PWD/models/hybrid/cache/hf"
$env:GRADIO_TEMP_DIR = "$PWD/models/hybrid/cache/gradio-studio"
$env:MPLCONFIGDIR = "$PWD/models/hybrid/cache/matplotlib"
$env:PYTHONDONTWRITEBYTECODE = '1'
uv run sightguard ui --research --port 7861
```

访问 http://127.0.0.1:7861，选择 Annotation Studio。
默认数据集 `real_world/annotations/phase3/manifest.csv`，八段视频均为 development，
初始没有已完成人工审核的真值。选择 video_id 后点击“加载原始视频”。
“上一段/下一段”在当前已加载的数据集内移动；手动更换数据集后须再次点击加载。
没有原视频的旧合成 manifest 不进入 Studio 选择列表，原有研究页面仍可使用它们。

## 精确 Viewer

预览组件直接解码 manifest 关联的原 MP4，只有原图，不显示检测框、track_id、
风险分数、模型事件或 Laya 输出。首先验证视频 SHA256、标签哈希及视频身份。
本批视频使用 Phase 6A 已保存的处理 PTS，并与完整原视频解码 PTS 比较，容差 1 ms；
没有对应处理记录时完整解码取得原 PTS。不根据固定 FPS 推算标注时间。

“上一帧/下一帧”、帧编号滑块、开头/结尾均按原帧索引定位。指定毫秒跳转选择不晚于
输入时刻的原帧，显示实际帧编号及实际 PTS。每次读取再次核对解码 PTS，避免图像与
时间错配。本批每帧约 50 ms；未来 VFR 仍按实际 PTS 选择。

“播放原始帧/暂停”采用 Gradio Timer 顺序显示帧，以相邻 PTS 差作为下一次计时间隔。
它是标注检查播放器，受浏览器和服务请求耗时影响，可能慢于原视频；不宣称 20 FPS
流畅播放或原速音视频同步。没有音轨；精确时间由选中原帧提供，与墙钟速度无关。

## 人工时间轴与表单

时间轴白线标记当前选中原帧。点击空白部分定位；点击条带选择对应事件/排除区间，
表单显示精确毫秒值。蓝色为事件，绿色为 initial 机会，橙色为 escalation 机会，
紫色为无法判断，灰色为排除。像素绘图只影响展示，不会把存储时间取整成秒。

1. 填写自己的 `annotator_id`，点击“新建事件”。
2. 播放或逐帧检查原画面，用“当前帧设为开始/结束”，也可数值微调。
3. 填写唯一事件 ID、情境类型、人工最高等级、证据状态和 COCO class_ids。
   例如人 0、自行车 1、汽车 2、摩托车 3；类别来自人工画面观察，不引用检测器结果。
4. 编辑机会表：`opportunity_id`、`kind`、`window_start_ms`、`deadline_ms`、
   `minimum_severity`。确认事件恰有一个 initial；升级需晚于首次起点且最低等级更高。
   “添加首次/升级机会行”只创建待编辑表单行，必须人工确认时间和等级后应用。
5. 需要空间匹配时添加人工目标区域行，填写区域生效区间及归一化 xyxy。坐标由人工
   根据当前原画面填写，本版不提供拖拽画框；不读取模型框或跟踪 ID。默认空坐标不能保存。
6. 点击“应用区间编辑到草稿”，通过现有 Schema 与重叠校验后更新人工时间轴。
   这一步只改会话草稿，还没有写盘。
7. 点击“保存 Draft”才写盘。重新加载后点击时间轴可恢复已保存区间及机会/区域表。
   “删除选中区间”只删除会话草稿中的该区间，需要再次保存。

同类别重叠确认事件需要独立目标区域；机会必须落在事件内，截止点必须早于视频的
排他结束时刻。排除区间为正时长且不超出视频。无效表单不会覆盖有效存储。

证据不足时优先“新建排除/无法判断区间”，选择 uncertain 并填写原因，不必选择
风险等级。它映射为 Schema 3.0 `excluded_intervals`，原因以 `uncertain:` 标识。
也支持 Schema 中 uncertain/excluded 事件，但其必填 severity 是结构性占位，
不作为风险真值匹配，不能添加提醒机会。不能把不可见或未标注当作无风险。

## 保存、审核与恢复

- **Pending**：尚无 Studio 人工保存记录、作者 UNASSIGNED。
- **Draft**：显式人工保存，核心文件仍为 `annotation_status=draft`。
- **Ready for Review**：初次标注完成；核心文件仍为 draft，不能做监督 benchmark。
- **Reviewed**：对未更改的 ready 文件，由第二个不同 ID 的审核者显式确认；核心文件
  才变为 reviewed。标签/manifest 哈希绑定由保存事务更新，继续兼容 seal-labels。
- **Invalid**：标签、来源、时间或审核证据校验失败；不会视作负样本。

Ready 和 Reviewed 均要求显式勾选“已完整查看原始视频”。如果 events 为空，还须
勾选“人工明确确认没有值得提醒的事件”。该勾选是人的声明，程序不能证明观看或
独立性。不得通过点击按钮代替真实原视频审核。审核规范见
[annotation_guidelines.md](annotation_guidelines.md)。

整段都无法观察时，可用有明确原因的 excluded_intervals 覆盖完整视频，审核后作为
全排除数据；不要求勾选无风险，也不能解释为负样本。只排除部分时，其余可评价
部分没有事件仍须人工确认，不可因没有标注而自动声明无风险。

审核状态、时间、原视频哈希、第二审核者和分歧存入相邻 `.review.json` sidecar，
不增加或改变 Schema 3.0 核心字段。所有保存前的标签、审核文件及 manifest 以 UUID
保存在 `annotations/studio_history/`，不会覆盖第一次证据。建议第二位人员先独立看
原视频、留独立记录，再对照第一次标注；本版没有用户权限系统或盲化身份认证。

Reviewed 和 test 数据只读，不能从 Studio 反向修改已审核结果。如需修订，应创建
独立数据集副本并重新走审核/冻结协议流程，本版没有一键版本分叉功能。

文件写入先全部校验和准备临时文件，再逐文件原子替换；可捕获的写盘异常恢复原标签
及 manifest。保存前比较标签、审核记录和 manifest 的组合哈希，拒绝过期会话的
并发写入。切换视频/数据集加载、选其他区间前，未应用/未保存编辑会被拦截。
要放弃编辑，明确勾选丢弃再加载；不会自动保存。

多个文件的更新不是数据库事务：突然断电/进程被强杀发生在替换之间，可能留下哈希
不一致。此时加载/验证拒绝该记录，需从 `studio_history` 恢复标签与 manifest，
再重新验证，不得直接忽略哈希错误。浏览器刷新/关闭会丢失未保存会话草稿，当前
不提供 beforeunload 拦截或自动恢复；请显式保存后再刷新。

## Visual Attributes

展开独立子面板，加载现有关键帧任务；验证图像哈希及原视频身份。可以填写可见物体
类别、可见性、左/中/右、可见候选通行区重叠、遮挡、画面运动及证据充分性。
unknown 保存为 null，不能解释为 false。画面运动须配合相邻原帧观察。

显式保存到当前数据集标签旁的 `<video_id>.visual.json`，重新加载任务可恢复已保存
属性。它不会修改 Phase 6A 原任务清单，不写风险事件 JSON，也不运行 Laya。
测试数据集副本的属性与原始真实数据严格分开；更改旧视觉标签同样保留历史副本。

## 验证、进度和 Benchmark

“验证当前草稿 Schema 3.0”复用原 Schema 和同类重叠验证，报告具体字段/时间问题。
“检查正式 Benchmark 闸门”直接调用既有 `load_dataset(require_reviewed=True)`，
不绕过草稿拒绝，不计算准确率。“刷新进度”显示总量、pending/draft/ready/reviewed/
invalid 及逐视频状态，reviewed 的 Benchmark 可用性还须通过原数据验证。

在审核真正完成后，继续使用原 CLI：

```powershell
uv run sightguard seal-labels --dataset datasets/real_world/annotations/phase3/manifest.csv
uv run sightguard validate-labels --dataset datasets/real_world/annotations/phase3/manifest.csv
uv run sightguard benchmark --dataset datasets/real_world/annotations/phase3/manifest.csv --split development --data-kind real --config configs/benchmark.yaml --risk-config configs/risk.yaml --output-dir outputs/real_world/reviewed_benchmark_v01
```

一部分已审核时，建立仅含这些样本的新 manifest；不能把其余草稿批量标为 reviewed。
Studio 没有新 Benchmark 实现。模型结果只在独立 Analysis/研究页查看，标注阶段请
不要查看这些结果；UI 无法证明人员未看过其他页面。

## 实现边界

`ui/annotation_store.py` 负责已登记数据集路径限制、PTS、Schema、事务与审核记录；
`ui/annotation_tab.py` 负责原始图像、人工时间轴、表单和会话修改保护。
浏览器只提交注册 manifest ID/video_id，不能指定任意文件写入路径。
媒体使用内存原图及已有 Gradio 缓存，未开放整个 datasets 为任意下载目录。
核心风险和评价模块不依赖 UI，新工具复用其既有校验。
