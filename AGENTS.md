# Home-Seek

DeepSeek-V4-Flash 单卡 RTX 4090 推理引擎。

## 纪律

1. 修 bug 必须先写复现该 bug 的快速单元测试 (L1), 再修代码
2. 功能里程碑完成后，或实现某个性能优化后，必须 `make profile`, 对照 `.agent_memory.md` 基线检查
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
- 实施记录：`docs/implementation_notes.md`，记录实际实现中与设计相悖的地方，未预见的约束（硬件、库限制），尝试过但失败的方案。
- 里程碑记忆: `.agent_memory.md`，profile 基线 + 瓶颈分解 + 下一步推荐

## 缓存体系 (当前)

```
请求 expert (layer, eid)
  ├─ 1. _gpu_hot_experts (GPU BF16, ~25 条 × 48MB, 永久缓存)
  ├─ 2. _gpu_bf16_cache  (GPU BF16 LRU, ~76 条 × 48MB, 自动淘汰)
  ├─ 3. ExpertWeightCache (CPU FP4, 5120 条 pinned+unpinned, pin=永不淘汰)
  ├─ 4. _gpu_expert_store (GPU FP4, 128 条, 命中率~0%)
  └─ 5. safetensors mmap (RAID 1.5 GB/s)
```

## 性能评估铁律

- **必须用 `--temperature 0` (argmax)** 做 A/B 对比, 否则 MoE 路由噪声 (±15%) 淹没真实收益
- **必须 3+ 次重复取平均**: 共享磁盘负载波动导致吞吐 ±2-3%
- **关键指标**: 吞吐 (t/s) + File loads (确定性, 唯一可靠对比指标)
- 修 bug 先写 L1 测试再修代码

## 已知陷阱 (gotchas)

- `ExpertWeightCache.clear()` 原实现清空 pinned 条目 → 任何预加载机制必须修复或检查此行为
- 共享 RAID 多线程读盘反效果 (I/O 争用), prefetch 默认禁用
- `tl.load/store` 必须有行列掩码; `tl.dot` 要求 M,N,K≥16
- M=1 decode 时 fused Triton kernel 比 cuBLAS 慢 8× (15/16 SM 空转)
- 修改 Triton kernel 后删 `~/.triton/cache/`
- CPU 全量预载 (11008 专家, 169GB RAM) 反效果 — 挤占 page cache, 推理 +8%

## Triton Kernel 铁律

- `tl.load` / `tl.store` 必须有行列掩码 (M < BM 时必加行掩码)
- `tl.dot` 最小: M≥16, N≥16, K≥16
- 修改 kernel 后删 `~/.triton/cache/`
- 不得在 `@triton.jit` 内定义嵌套函数
