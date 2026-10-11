# Corridor Intelligence v1 验证与阶段任务报告

日期：2026-10-11。完成范围为现有 Spatial Grid v03 的候选带证据评价、离线消融、真实浏览器检查和基线回归。本阶段没有输出导航动作或改变提醒策略。

## 实验版本、输入与溯源

分支 `codex/corridor-intelligence-v1` 从 `25a9ab631d97aac3f26233becbe76c81b9d74fa5` 创建。主实验执行时 HEAD 仍为该父提交，新增实现的准确文件 SHA256 已保存在 `outputs/corridor_intelligence/v01/run_metadata.json.source_sha256`，不能只用父 HEAD 当作完整实验代码版本。实现与本报告一同以 `experiment: add evidence-based corridor evaluation` 本地提交；精确提交 ID 由最终回复及 `git log -1 --format=%H --grep="experiment: add evidence-based corridor evaluation"` 获取。按本任务要求未推送、未合并 master。

- 正式输入：`outputs/spatial_grid/v03`，8 段 UrbanEgo、每段30帧，共240采样帧；帧0–290步长10，视频 PTS 0–14500毫秒步长500（2 Hz），原图1280×720，15×15=225格。
- 原网格 ObjectGridRelation 共1,123条；本阶段三种含检测消融的候选交叉关系共3,669行，不是3,669个独立目标。960条 corridor assessment、2,880条候选评价，每组720个几何模板。
- Manifest SHA256：`b4de54f10c7236968532792a05ad1997b4033f0f7c9ed9d5144421185bcb1d42`。
- Corridor YAML SHA256：`f4c8e78320d022bd4231877827f22a5729aaa2f150f4d6ac791bcee9ac41d409`。
- Spatial v03 根 metadata SHA256：`3f19d257f695007adbee6e1196989cfdf3355ecd1672015e26f25501f00050b1`。
- uv.lock SHA256：`a95c90b4571d20df3642b6f8887f8c2ae7f4a00333ff787a6a5e6d96c2962637`；无新增依赖或模型。
- 继承 Depth Anything V2 Small revision `5426e4f0f36572d16453bbda7a8389317b1bef99`，checkpoint SHA256 `3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70`；继承 SegFormer checkpoint SHA256 `8ffa8fdac140ffaef9119150aca65e9fcb4e7a165e60b80be0102cc805d73a9c`。本轮没有加载或重新推理这些模型，继承缓存对应模型元数据哈希也在预检中验证。

每视频空间帧文件 SHA256：

```text
urbanego_run8_01: 6fbb17d2e95291bb158c6e10cedaa7a9d82c37cece67dc3f9391491fe3ea21fb
urbanego_run8_02: 751f0ad8864e622aba4e1477c4b860a0a8fe5d3c41f76dddc1b9ca801e65a283
urbanego_run8_03: fc7985ee6623b7913a66a6d545cdc9576f70a3e81c2082cc3462671dc5a15ff9
urbanego_run8_04: 09582d916ae3117b410c6032eb9bdf663e3d41e3b33c2de418acb27bc5af1814
urbanego_run8_05: fb179d35ef0e4a26170bd7050dd1156ecf303c722462439ff528a14db9c9eaa9
urbanego_run8_06: 27d1b3df78ba736664a21dbab8bcd8cd5813543ed1b3779243255521b7c17d1a
urbanego_run8_07: e25b4832bb59bb2e85d9aabdedf56307f3df921458365c74646506d47aaa5908
urbanego_run8_08: f3d09d9c3f2c9af7991814e8eb502f3451454f5deaf0e08453bcde390e08ae6b
```

每视频 parent_input_sha256 还完整记录 tracks、semantic/depth JSONL 与源视频；逐帧 sources 保存 semantic/depth NPZ 引用、哈希和原质量原因。预检核对源尺寸、原视频身份、真实 PTS、原生产代码、全部225格及单独 object_relations 与 frame 内容一致性。tracks v1 行没有 video_id，以已核验的视频目录、manifest、文件哈希及每行尺寸/PTS绑定会话，未补造字段。

## 八段真实回放结果（实际程序输出）

A、D 每视频均0个可评价候选、30帧ABSTAIN。B/C共享相同基础门禁，下面可评价数相同；列 C 状态依次为 BEST_SUPPORTED / TIED / ABSTAIN，UNKNOWN 为720个候选内单元 exposure 的逐视频均值，不是安全概率。

| 视频 | 采样帧 | B/C可评价候选 | C帧状态 | C未知 exposure均值 | B→C排序改变帧 |
| --- | --- | --- | --- | --- | --- |
| 01 | 30 | 89/90 | 9 / 21 / 0 | 0.0660 | 0 |
| 02 | 30 | 82/90 | 25 / 4 / 1 | 0.3054 | 1 |
| 03 | 30 | 76/90 | 27 / 3 / 0 | 0.3622 | 0 |
| 04 | 30 | 50/90 | 26 / 3 / 1 | 0.6846 | 0 |
| 05 | 30 | 86/90 | 19 / 11 / 0 | 0.1959 | 1 |
| 06 | 30 | 83/90 | 24 / 5 / 1 | 0.3027 | 1 |
| 07 | 30 | 88/90 | 19 / 11 / 0 | 0.2326 | 3 |
| 08 | 30 | 82/90 | 11 / 18 / 1 | 0.1676 | 3 |

总计：

- A：0/720可评价；240 ABSTAIN。几何生成没有提供通行证据。
- B：636/720可评价（88.33%）；158 BEST_SUPPORTED、78 TIED、4 ABSTAIN；eligible成本均值0.201344。
- C：636/720可评价（88.33%）；160 BEST_SUPPORTED、76 TIED、4 ABSTAIN；eligible成本均值0.218826。平均深度加权贡献0.018545（包含未通过门禁候选的可用分项）。
- D：0/720可评价；240 ABSTAIN。720个候选全部缺少实测姿态。另有179个候选过未知门禁、218个过语义置信度门禁、91个过冲突门禁、90个过下部框覆盖门禁；原因可同时发生，不能相加当独立帧数。
- B/C地面门禁原因：81个支持不足、36个底行不足、36个连续性不足、6个语义源不可用；同样是可重叠候选原因。全帧ABSTAIN是V02/10、V04/200、V06/150、V08/290。
- C/D未知均值0.289622344；B为0.289622283，差异来自极少数无效深度像素，而不是新增道路风险标注。D的姿态未知独立记录，没有隐去或把它分摊成像素概率。
- 232对相邻500ms样本中，B唯一支持候选身份（包括null）变化103次，C101次；D为0是全部弃权的自然结果。未验证跨帧尺度/几何身份，不能宣称稳定性提高或D更稳定。

## Depth 的真实差异与不变之处

B/C成本发生变化的帧236/240，排序变化9帧，状态变化10帧（不只是净增加2帧唯一支持）。4帧因基础证据不足没有可比较 eligible 成本。B→C新增深度项，同时启用来自深度变化的冲突理由；完整逐候选 component delta 已保存，不能把所有成本增量都归因于 near-depth 项。

具体程序输出：

1. **V07 frame220，PTS11000ms**：CENTER成本B=0.142298、C=0.189285；深度贡献0.043616、深度冲突新增0.003371，下部相对近深度代理0.333017。RIGHT成本B=0.150781、C=0.166634；深度贡献0.011464、冲突新增0.004389，下部代理0.062873。排序由CENTER→RIGHT变为RIGHT→CENTER，但成本差0.022651仍小于0.03，B和C都TIED，best=null，D因姿态未知ABSTAIN。这是深度确实改变评价的案例，不能宣称右边安全。
2. **V02 frame60，PTS3000ms**：CENTER成本0.400849→0.462652，新增深度0.055314及冲突0.006489；LEFT 0.410647→0.458365。第二、第三名交换，RIGHT保持最低成本。框覆盖不同的多目标证据也保留，未强制更改正式动作。
3. **V05 frame120，PTS6000ms**：LEFT 0.128495→0.133742，CENTER 0.130434→0.133664；最低排序发生微小交换，但两组都TIED。near-depth代理为0，差异主要来自格内深度变化及深度冲突，不能解释为检测到了新的近障碍。
4. **V08 frame80/90**：地面纹理的变化项产生约0.0033的微小差异，CENTER/RIGHT的排序改变，仍TIED、best=null。这说明排序变化不自动意味着实际通行意义；没有GT支持“Depth改善了方向正确率”。

## V03 / V04 / V05 / V07 / V08 困难场景

- **V03**：停车车辆与道路/人行空间混合。程序C仍有76/90可评价模板，不能把这当作可通行标注；代理复核frame150可见大量车辆遮挡与地面未知，D保留弃权。
- **V04**：全视频候选未知均值0.6846，基础可评价候选仅50/90。frame200的CENTER ground=0.174746<0.2，unknown=0.687955；B/C均ABSTAIN，三个候选都没有强制排名。其他29帧B/C不是全部弃权，不能夸大成整个V04都因地面不足无判断；D所有帧有姿态等额外拒绝原因。
- **V05 frame290**：LEFT/CENTER/RIGHT框代理0.0123/0.1064/0.1835，分别保留1/5/6条交叉对象关系，含不同chair/person track ID；覆盖采取每格max，关系保留每个原对象。左带只说明图像地面证据较强，未输出绕桌椅导航。
- **V07 frame90**：LEFT/CENTER/RIGHT框代理0.0827/0.3547/0.2820，分别3/6/5条对象关系。CENTER下部深度贡献约0.0110，LEFT约0.0030。原分类含tv等未经独立确认的标签，不能通过画面印象改写已保存检测。
- **V08 frame290**：原语义源UNKNOWN，B/C基础门禁拒绝所有候选，ABSTAIN；有原始地面RGB纹理和有效相对深度也没有强制补出地面支持。

上面的数值属于程序输出。八视频代表帧 contact sheet 及RGB页面由代理目视复核：可见车辆遮挡、行人/餐桌椅拥挤、低头地面纹理等；这是代理目视观察，不是额外人类评审、instance 标注或通行 Ground Truth。本阶段没有证明候选方向正确率或真实碰撞效果。

## 实测性能与资源

Python3.12.15、NumPy2.5.3、OpenCV4.14.0，Windows11。本次 CPU 离线评价不加载PyTorch/YOLO，GPU没有承担本阶段推理。下表每项按同一源帧的四组消融合计计时，候选生成一次共享；algorithm_total包含生成、证据聚合、门禁成本、对象关系、排名/溯源，排除加载、序列化与预览。

| 阶段 | P50 ms | P95 ms |
| --- | --- | --- |
| algorithm_total | 3.1203 | 3.6155 |
| candidate_evaluation | 0.0892 | 0.1049 |
| candidate_generation | 0.0858 | 0.1050 |
| evidence_aggregation | 2.3860 | 2.6775 |
| full_sample | 53.2372 | 86.6233 |
| load_validate | 4.7530 | 5.9218 |
| object_relation | 0.3743 | 0.7635 |
| ranking_and_provenance | 0.1693 | 0.2228 |
| serialization | 2.5456 | 3.3508 |
| visualization | 42.3362 | 75.1704 |

预检2.6349秒；根 wall 19.1869秒，包括预检、240帧、HTML/JS生成及输出哈希，排除最后metadata写入。进程CPU时间84.5469秒，OpenCV可能使用多线程，因此不能当单核利用率或每帧推理时间。峰值working set 240.99MiB，记录时working set 171.11MiB。预览解码/保存是主要耗时，不据此宣称实时摄像头部署已验证。240条原始阶段计时保存在 timings.jsonl。

## 自动化、真实集成与浏览器

- 最终 `ruff check .`通过；全仓库pytest **508 passed, 18 skipped**。新增默认corridor测试51项，覆盖配置/完整边界与面积、无地面/连通中断、类别分离、重复框多track、nullable身份、近远深度、缺测/退化深度、有效像素门禁、冲突、来源/时间/视频不一致、排序并列、消融无泄漏、确定性及无模型/风险/音频依赖。旧有效测试未删除或弱化。
- 显式开启本轮 saved corridor 与既有 saved spatial 集成：**8 passed, 1 skipped**；跳过的是未启用的真实模型重推理集成，本轮无需重跑模型。真实八视频完整预检与另一个临时目录重复评价逐行比较均通过。
- `scripts/validate_corridor.py`核验240源帧、960 assessment、2,880候选及3,669对象交叉行；输出source/code/output哈希及分项求和、null成本、并列best=null、ABCD同键同几何均通过。
- 独立重放原YOLO/BoT-SORT tracks产生 **2,400风险帧、257提醒**；risk_frames、events、alerts逐行与原baseline相等；**1,954冻结文件SHA256全部不变**。本轮没有调用音频适配器，没有替换正式事件。
- 使用computer-use浏览器控制，经`http://127.0.0.1:8766`真实加载页面，测试帧0/10/290、候选切换、A/B/C/D、RGB/未知层、点击cell100明细及对象关系；最终版本又核验36点精确阶梯边界、V04/200的B弃权、V07/220的C并列及A/D弃权。图片naturalWidth=1280；检查未见console warning/error。截图与检查记录在`outputs/corridor_validation/v02`，未用Node静态模拟代替真实浏览器。
- 原7860/7861服务未停止；本轮独立只读viewer使用8766。没有通过file URL绕过先前策略限制；本任务明确要求HTTP真实浏览器检查。

## 产物、失败记录与复现

主结果`outputs/corridor_intelligence/v01/`：每视频三种JSONL、summary/run_metadata、30张RGB和30张诊断预览、evidence.js/index.html；根 ablation_summary、failure_cases（240帧默认D弃权）、timings、run_metadata、validation及输入/源/输出哈希。回归重放与浏览器证据在`outputs/corridor_validation/v02/`，八视频contact sheet为`review_contact_sheet.jpg`；冻结快照在`outputs/corridor_audit/v01/frozen_hashes.json`。

首次预检因tracks v1没有video_id而失败，原失败metadata保留在`failed_v01_preflight/`；通过hash绑定原会话后修正读取逻辑，未更改旧数据。期间有效像素覆盖、阶梯边界和最终计时范围修订前的探索结果保留在独立`exploratory_v01_*`目录。报告所有主指标只来自最终v01，不混用探索结果。生成视频/数组/结果全部受Git忽略，不提交模型或媒体。

```powershell
# 全部目录须不存在；已有 v01 时用自己的新目录名
uv run --no-sync python scripts/run_corridor.py --output outputs/corridor_intelligence/reproduction
uv run --no-sync python scripts/validate_corridor.py --corridor outputs/corridor_intelligence/reproduction --output outputs/corridor_validation/reproduction
uv run --no-sync ruff check .
uv run --no-sync pytest -q
$env:SIGHTGUARD_RUN_CORRIDOR_SAVED_INTEGRATION='1'
$env:SIGHTGUARD_RUN_SPATIAL_SAVED_INTEGRATION='1'
uv run --no-sync pytest tests/test_corridor_integration.py tests/test_spatial_grid_integration.py -q
# 浏览器打开 http://127.0.0.1:8766/ 后选择视频
uv run --no-sync python -m http.server 8766 --bind 127.0.0.1 --directory outputs/corridor_intelligence/reproduction
```

需已有本地UrbanEgo与父阶段数据，来源/生成方法见原Depth及Spatial Grid文档。集成测试逐行对照本机最终v01，实际运行记录仍含父HEAD+源码哈希。统计并列门限与权重未基于此批样本调优。

## 最终验收20项答复

1. 新增候选schema/generator/adapter/evaluator、对象关系、CPU runner、浏览器viewer、验证与测试接口，详见实现说明。
2. 参考Roana实际fusion、planner、state machine、MotionQuality及架构/状态文档，固定上游revision。
3. 没有直接复用、翻译或改写第三方源码、测试或fixture。
4. 确定性插值生成三个逐行加宽图像带，默认各36格，ABCD共享几何。
5. 空间含义是原图像素观察区域，底部是关注区，不是米制路线或近场安全通道。
6. Cost由ground、continuity、obstacle、depth、unknown、conflict六项显式加权；无资格则总成本null。
7. 地面描述支持/类别/组件连续性；检测描述多框覆盖代理；深度提供帧内近远和变化证据，不提供碰撞概率。
8. UNKNOWN保存缺测与姿态，基础/严格门禁产生ABSTAIN；并列不强制唯一best。
9. 原ObjectGridRelation与候选单元相交，保留对象身份、覆盖、地面/深度/背景对比与来源；event_reference=null。
10. 八段240帧真实保存输入全部完成，B/C共636可评价候选，默认D全部弃权。
11. B→C成本236帧变化、排序9帧变化、状态10帧变化；D增加门禁而不是改变正式提醒。
12. Depth确实改变数值与部分排名；V07/220、V02/60、V05/120提供分项增量与未强制选择的案例。
13. V04/200在B/C确因地面支持不足弃权；其余V04帧不夸大成同样失败，D另受姿态门禁。
14. V05/290和V07/90多个对象分别影响候选框覆盖与相对深度，关系不合并为一个目标。
15. 系统未输出真实方向动作，TIED/ABSTAIN的best均null；无通行GT，不能断言研究best从未选错。
16. 四消融算法总计P50/P95=3.1203/3.6155ms；含加载/序列化/预览每源帧53.2372/86.6233ms，批量wall19.1869秒、峰值240.99MiB。
17. 正式risk/events/alerts独立回归完全不变，共257提醒；没有音频调用。
18. 默认508通过18跳过，saved真实集成8通过1跳过；有完整八视频验证、逐行重放和真实HTTP浏览器检查。
19. 分支与本地提交消息见本文开头；准确提交ID由最终回复和Git记录提供，未推送。
20. 下一阶段首先需要实测姿态/标定、尺度/足迹、instance接地与跨帧身份、可靠RiskEvent对应和独立风险/通行GT；再研究时间策略与保守决策，本轮不提前实施。

程序输出、代理目视观察与未经过GT验证的结论已分别标明。本阶段完成的是证据解释与保守弃权，不是找到了安全路线。
