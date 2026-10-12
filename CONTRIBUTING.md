# 参与 SightGuard

## 开始前

阅读 [AGENTS.md](AGENTS.md) 和[架构](docs/architecture.md)。保持修改小而可回退；在独立分支工作，不覆盖其他工作区的未提交内容。用户体验、模型改进和研究报告都欢迎通过清晰的 PR 提交。

## 本地开发

PC 使用 Python 3.12、uv 和已提交的 `uv.lock`：

```powershell
uv sync --locked
uv run ruff check .
uv run pytest -q
```

Android 环境与模型准备见[开发指南](docs/development/android.md)。默认测试不应下载数据或依赖 GPU；真实模型和设备测试单独运行。

文档修改后运行：

```powershell
python scripts/check_docs.py
```

## 提交内容

说明用户可见的变化、适用条件和实际检查结果。不要把合成测试、公开数据推理、模拟器与实体设备证据混在一起；不能完成的检查明确保留。测试和详细实验记录进入对应文档，不把过程日志堆入项目主页。

代码使用英文命名、明确类型和可序列化接口。感知模块输出独立数据结构；风险和事件核心不引用模型/UI 对象。时间逻辑保留源帧来源和会话隔离。

不要提交模型、原始视频、个人截图、设备原始日志、密钥、APK 或临时环境。不要为通过检查删除有效测试、修改固定真值或降低时效门禁。第三方资产保留来源和许可，见[第三方说明](THIRD_PARTY_NOTICES.md)。
