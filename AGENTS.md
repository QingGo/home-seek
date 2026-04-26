# Home-Seek 开发指南

## 工具链
- **依赖管理**: `uv` (非 pip)
- **环境**: 激活 `.venv` (`source .venv/bin/activate`) 或用 `.venv/bin/python`
- **静态检查**: `ruff check home_seek/ tests/`
- **测试**: `python -m pytest tests/ -m "not slow" -v --durations=10` (快速), `make integration-tests` (慢)
- **入口**: `main.py` → `home_seek.inference_engine:main`

## 测试分层
- **单元测试** (`tests/test_*.py`): <1s/个, 标记 `@pytest.mark.fast`。应该尽量覆盖所有主要路径。
- **慢测试**: 标记 `@pytest.mark.slow`, 不频繁运行
- **集成测试** (`tests/integration/`): 需要真实权重, 标记 `@pytest.mark.integration`
- 集成测试/慢测试的逻辑应尽可能被快速的单元测试覆盖; 集成测试仅验证端到端通路。

## 调试规范
- 使用 `self._log(msg)` 而非 `print`; 通过 `verbose=True` 控制
- CUDA 错误用 `CUDA_LAUNCH_BLOCKING=1` 定位
- 混合精度: 保持 BF16 主路径, FP32 仅用于中间计算

## 关键约束
- 目标: 单卡 RTX 4090 (24GB), FP4/FP8 量化权重
- 权重路径: `weights/` (包含 46 个 safetensors 分片)
- FP4 原始数据保留在 CPU; 仅热缓存 (BF16) 常驻 GPU
- MoE: 256 专家, top-6, 共享专家按需加载并缓存

## 常见陷阱
- TileKernels 首次调用触发 JIT 编译 (~100ms-2s), 须预热
- `torch.unique` 在 CUDA 上是同步点, 避免在热路径使用
- `reduce_fused` 要求 `topk_weights` 为 float32

## 结构化状态与记忆管理 (Structured Memory)
为了防止在长对话中丢失上下文，本项目采用**短期会话记忆 + 长期架构记忆**分离机制，统一维护 `.agent_memory.md`：
- 短期记忆：当前会话临时任务、即时踩坑、未闭环问题、会话级上下文
- 长期记忆：核心架构决策、人类最终决策、通用踩坑沉淀
- 文件位置：根目录 `.agent_memory.md`（已加入 `.gitignore`，纯本地使用）