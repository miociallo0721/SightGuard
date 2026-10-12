> 历史资料：保留当时的实现与验证记录。当前使用入口见仓库 README；本文不是当前能力清单。

# Android 0.3 风险相关性政策

PC `configs/default.yaml` 默认只检测 COCO 0/1/3；Android 检测全部80类。
两端核心评分不使用类别 importance；不存在 unknown class 默认高重要性参数。
核心使用关注区、框面积/底部位置及可靠图像运动，不表示距离或碰撞概率。
没有运动时空间/外观权重重新归一化：关注区位置贡献0.45/0.8=0.5625，
面积达到0.20且底边达到0.95的框 rawScore=1；静止、运动可用时仍能达到0.80。
所有类别直接进入该公式是书本/鼠标等获得HIGH的主要原因。
初帧缺少运动并不是低风险证据，重新归一化还会先产生HIGH，之后即使静态分数降至
MEDIUM区间，也需现有降级迟滞hold400ms后才变为MEDIUM。
可复现场景：1000x1000，框(400,200,500,600)，初帧0ms score=.684326923076923/HIGH；
静止300ms score=.5474615384615384但level仍HIGH；690ms仍HIGH，700ms才MEDIUM。
因此测试“普通MEDIUM冷却”的输入必须在初始无motion与之后可靠静态两种公式下
都真实落入MEDIUM；不能按第二帧rawScore推测level或只修改UI等级。
原界面只展示当前ACTIVE事件的最大score，并使用固定红色；核心并未把全部事件设为HIGH。

Android新增 `Tracking -> RiskRelevance -> AssessmentEngine`，核心、PC配置和parity不变。
`VisionResult.raw` 与 `records`保留完整原始类别、置信度和框；仅engine收到过滤副本。
风险observation索引属于过滤输入；`relevance.detectionIndex`属于完整records输入。
RelevanceConfig集中保存版本、类别集合、图像位置阈值和源时间确认参数。
Primary包含行人、车辆和相关动物；Conditional包含椅子56、餐桌60、行李箱28等
家具/可能阻碍移动的物体。Conditional需focus>=.8、area>=.04、bottom>=.65，
至少3次观察且持续500ms，间隔不超过500ms。这些是可解释的图像代理政策，
目前通过合成场景验证，不能声称已校准真实行走危险。类别变化、丢失或位置失效重新确认。
默认Contextual（含书73、鼠标64、键盘66、电脑63、刀43）不生成自动步行风险。
墙/台阶不属于COCO类，额外空间模型仍只提供shadow证据。

刀/剪刀增加时间和置信度可信度标记。未稳定或confidence<.65时标记类别不确定；
原始类别不被改写。稳定的knife也不会因类别直接生成步行风险。
COCO无pen，本轮没有训练或模型层面解决pen/knife混淆。
原始记录可用于复现和后续混淆样本研究。

等级仍由核心LOW/MEDIUM/HIGH(.35/.65)生成；升级margin .02，降级margin .05，
降级hold 400ms，事件确认250ms/2观察。NONE是没有当前已确认事件的显示状态，
不能将UNKNOWN/UNAVAILABLE等感知质量解释为安全。
LOW默认静默、MEDIUM首次提示、所有真实HIGH可越过普通播报冷却。
新HIGH仍须经过原生事件确认，播报在onStart后提交，几何去重和过期取消保留。

自动化回归新增 `RiskRelevanceTest` 和 `AlertPresenterLevelsTest`：
书/鼠标/键盘/电脑/knife、COCO条件类别、位置/时间边界、重复时间戳、丢失/类别变化、
重置、配置错误、原核心分数边界/迟滞、真实CameraX冻结检测trace逐帧core输出一致、
无效几何LIMITED与推理失败UNAVAILABLE。
播报测试由真实AssessmentEngine计算LOW/MEDIUM/HIGH，覆盖LOW静默、
MEDIUM首次/普通冷却、不同HIGH行人/自行车越过普通冷却、接受后不重复、
同事件升级，以及只合并当前已确认且具备first资格的同方向行人。
这些是JVM政策回归；设备TTS抢占、实际onStart、音频/生命周期验收需独立设备验证。

主界面对刀类固定加“类别不确定”的保守标签，可信度状态仍保留在relevance记录中；不把稳定性当作类别正确性的证明。
自动启动状态与偏好、权限、主动停止、输入源及前台生命周期独立管理；首次用途说明后系统授权，默认开启前台自动感知，可关闭并持久保存。
文件模式/URI/位置和主动停止会随Activity保存；异步释放结束前的重启需等待旧主模型和空间会话释放。

迟滞回归保持相邻帧至少10ms；1ms间隔低于PC motionMinIntervalMs，会使运动证据不可用，不能被当作正常持续帧。
