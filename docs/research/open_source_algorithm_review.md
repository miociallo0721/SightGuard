# 开源算法借鉴记录：事件关联与可通行区域

检查日期：2026-10-11。只阅读与目标身份维护和障碍/通行区处理直接相关的源码，未克隆仓库，未复制代码。

## CaBot People

- 仓库：[CMU-cabot/cabot-people](https://github.com/CMU-cabot/cabot-people)，检查 `main` 提交
  [`4de04120e6d8f833b41e10c514938b80cc0f5d7a`](https://github.com/CMU-cabot/cabot-people/tree/4de04120e6d8f833b41e10c514938b80cc0f5d7a)，MIT。
- 项目实现：[`tracker_sort_3d.py`](https://github.com/CMU-cabot/cabot-people/blob/4de04120e6d8f833b41e10c514938b80cc0f5d7a/track_people_py/track_people_py/track_utils/tracker_sort_3d.py)
  以鸟瞰 3D 中心和 Kalman 预测圆计算两两 IoU，再用 Hungarian assignment 做一对一匹配；低于 IoU 门限的匹配被拒绝，未匹配检测创建新 ID。失配轨迹保留到时间过期，达到最少观察数后再输出。
  [`track_abstract_people.py`](https://github.com/CMU-cabot/cabot-people/blob/4de04120e6d8f833b41e10c514938b80cc0f5d7a/track_people_py/track_people_py/track_abstract_people.py)
  还用 3D 空间哈希去除来自多个相机的近邻重复框。
- 第三方部分：线性分配来自 SciPy，Kalman Filter 来自 FilterPy；CaBot People 的封装、生命周期、鸟瞰圆重叠度计算与空间哈希属于仓库实现。
- 对 SightGuard 的启发：冲突时明确拒配、失配后保留有限恢复期、匹配对保持一对一。CaBot 的状态坐标和多摄像头去重依赖 3D 中心/鸟瞰空间；现有代码不是单目图像框上的可直接移植方案，且不是 BoT-SORT 重识别实现。
- 传感器边界：代码使用 `center3d`，上游示例连接 RealSense RGB/深度相机；该几何关联不能在无深度输入时照搬。

## Roana

- 仓库：[floatinglifeai/roana](https://github.com/floatinglifeai/roana)，检查 `main` 提交
  [`7da28b7dd127f002986711ea62249b73de723739`](https://github.com/floatinglifeai/roana/tree/7da28b7dd127f002986711ea62249b73de723739)，AGPL-3.0-or-later。
- 已实现机制：[`CorridorGridFusion.kt`](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/app/src/main/java/com/roana/app/CorridorGridFusion.kt)
  把深度图缩到 15×15 网格，对分数至少 0.35 的目标框覆盖区域写入障碍值 0.96；未发现按类别排除“无关提醒”的过滤器。
  [`CorridorPlanner.kt`](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/app/src/main/java/com/roana/app/CorridorPlanner.kt)
  从网格底部中央寻找安全单元路径，按近障、路径长度和方向规则给出 STOP/LEFT/STRAIGHT/RIGHT。状态记录在 [`STATUS.md`](https://github.com/floatinglifeai/roana/blob/7da28b7dd127f002986711ea62249b73de723739/STATUS.md) 中被列为已实现的 V0 corridor loop，不是 README 里的远期设想。
- 对 SightGuard 的启发：将目标框与单目深度网格的空间关系作为一个独立实验特征，可用于以后诊断目标是否覆盖候选通行区域。它解决的是走廊选择，不是 track/event 身份维护或首次提醒去重；本次不移植该导航策略。
- 传感器边界：接口接收图像检测与 Depth Anything 深度图，没有要求 LiDAR/双目输入；依赖移动端深度模型与 Android/QNN 部署能力。网格深度经图像归一化，不应解读为校准的米制距离。

## 复用与许可证

本次只记录思路，没有复用源码。CaBot People 的 MIT 许可允许在保留许可与版权声明的条件下复用；Roana 的 AGPL-3.0-or-later 对修改和网络服务分发有源码提供义务，任何代码复用须先做项目许可兼容审查。SightGuard 的实验实现保持本仓库原生代码与数据接口。
