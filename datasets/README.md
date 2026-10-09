# SightGuard 数据集

`manifest.csv` 是真实数据导入的空模板，空 manifest 会明确报错，不会作为无风险数据集。
`videos/` 与 `annotations/` 用于本地原始视频和人工标注（默认忽略 Git）；`synthetic/`
保留小型、确定性、可重建的模拟 JSONL/标注，不包含真实视频。

真实数据准备顺序：

1. 使用有合法使用权限的视频，保留原片；记录拍摄来源组，同一原片裁剪片段放在同一划分。
2. 先用 Phase 1 `analyze` 生成 tracks.jsonl 与成功的 run_metadata.json。
3. 导入视频引用，命令不会复制或公开原视频：

```powershell
uv run sightguard import-video --video D:/data/clip01.mp4 --tracks outputs/clip01/tracks.jsonl --dataset-dir datasets --video-id clip01 --source-group-id recording01 --scene sidewalk --split development
```

4. 标注者只看原视频，不看检测框、风险或提醒。编辑创建的 draft JSON，填写作者、
   notes、独立事件和提醒机会；真实数据的 `label_source` 改为 `human_raw_video`。
   复核结束才改为 `reviewed`。没有风险的已审核视频可以 `events: []`；draft 空列表不可评价。
5. 执行 `seal-labels --dataset datasets/manifest.csv` 绑定人工修改后的标注哈希，再执行
   `validate-labels --dataset datasets/manifest.csv`。校验文件身份、视频时长/FPS、感知来源及覆盖。
6. 只在 development 调参；最终配置、划分、标注和源码确定后 freeze-protocol，再运行 test。
   标注纠错必须留审核记录并作为新的数据集/实验版本，不能为了提高指标改 test 标注。

完整模板在 `annotation_template.json`；其中的示例事件不是任何真实视频的真值。
已确认事件需要一个首次机会，可选多个更高严重程度的升级机会。class_ids 为人工语义类别映射，
不是模型检测是否成功的证明；两个同时存在的同类目标须提供独立的归一化目标区域。
事件类型说明风险情境，当前模型不分类“横穿/障碍”等情境，评价身份依赖类别、区域与时间。
详细规则见 ../docs/annotation_guidelines.md。

合成场景由 `scripts/generate_phase3_fixtures.py` 的预先声明情节和窗口生成，不读取策略结果：
开发集 1 段；测试集 4 段共 13 秒，含静止目标、尺度增长升级、完整感知漏检与无风险场景。
video_path 留空；synthetic 的 video_sha256 是场景描述指纹，不是假装真实 MP4 的哈希。
不同场景的来源组仅用于软件分组检查，不能视为独立真实拍摄样本或统计显著性依据。
脚本只在 datasets/synthetic 中重建固定文件；正常研究中不得运行它去重写人工 test 标注。

当前缺少已审核的真实动态场景。应补采正常通行、横向经过、静止障碍、遮挡/重现、
尺度变化、相机运动及质量不足的视频，至少由独立标注者复核，再冻结测试集。
不需要在道路上主动制造危险，也不应公开无必要的人脸或身份信息。
