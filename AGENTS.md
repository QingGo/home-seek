# Home-Seek

DeepSeek-V4-Flash 单卡 RTX 4090 推理引擎。

## 纪律

1. 修 bug 必须先写复现该 bug 的快速单元测试 (L1), 再修代码
2. 功能里程碑完成后必须 `make profile`, 对照 `.agent_memory.md` 基线检查
3. 提交前必须通过 `make lint test-unit`

## 命令

```bash
make install           # 首次或依赖变更后
make lint              # ruff 静态检查
make test-unit         # 单元测试 (<15s)
make test-integration  # 集成测试 (需 weights/)
make profile           # 性能分析, 对照基线
make smoke             # 最小冒烟: 1 token 推理
```

所有命令内部使用 `uv run`, 无需手动激活 venv.

## 关键路径

- 模型权重: `weights/` (46 个 safetensors, ~150GB)
- 热专家: `hot_experts.json`
- 架构文档: `docs/arch_design.md`
- 实施记录：`docs/docs/implementation_notes.md`，记录实际实现中与设计相悖的地方，未预见的约束（硬件、库限制），尝试过但失败的方案。

## Triton Kernel 铁律

- `tl.load` / `tl.store` 必须有行列掩码 (M < BM 时必加行掩码)
- `tl.dot` 最小: M≥16, N≥16, K≥16
- 修改 kernel 后删 `~/.triton/cache/`
- 不得在 `@triton.jit` 内定义嵌套函数
