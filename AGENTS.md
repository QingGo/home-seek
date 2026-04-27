# Home-Seek

DeepSeek-V4-Flash 单卡 RTX 4090 推理引擎。

## 纪律

1. 修 bug 必须先写复现该 bug 的快速单元测试 (L1), 再修代码
2. 功能里程碑完成后，或实现某个性能优化后，必须 `make profile`, 对照 `.agent_memory.md` 基线检查，并确认推理结果无异常。
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

## 缓存体系 (当前, V16)

```
请求 expert (layer, eid)
  ├─ 1. _gpu_hot_experts (GPU BF16, ~64 条 × 48MB, per-layer set, LRU)
  ├─ 2. _gpu_bf16_cache  (GPU BF16 LRU, ~76 条 × 48MB, 自动淘汰)
  ├─ 3. ExpertWeightCache (CPU FP4, 5120 条 pinned+unpinned, pin=永不淘汰)
  │    ├─ ~2118 pinned (per-layer hot × 43 + hash × 3)
  │    ├─ ~256 pinnned (MTP experts)
  │    └─ ~2746 unpinned (LRU)
  ├─ 4. _gpu_expert_store (GPU FP4, 128 条, 命中率~0%)
  └─ 5. safetensors mmap (RAID 1.5 GB/s)

共享专家:
  └─ _shared_expert_weights (GPU, 43 层 FP8 preload, 1 GB)
       → 首次访问 lazily dequant 为 BF16 3-tuple

MTP 模块 (投机解码):
  ├─ _mtp_weights: 33 个非专家权重 (GPU BF16, 含 attention/FFN gate/MHC)
  └─ ExpertWeightCache: 256 专家 (CPU FP4 pinned, key="mtp_0_{eid}")
```

## 性能评估铁律

- **必须用 `--temperature 0` (argmax)** 做 A/B 对比, 否则 MoE 路由噪声 (±15%) 淹没真实收益
- **必须 3+ 次重复取平均**: 共享磁盘负载波动导致吞吐 ±2-3%
- **关键指标**: Decode throughput (t/s) + File loads (确定性, 唯一可靠对比指标)
- **Prefill/decode 分离**: 用 `decode_time_s` / `num_generated_tokens` 而不是 `total_time_s` 算真实生成速度. `make profile` 现在输出 "Decode throughput".
- **Shared expert 修改需谨慎**: cuBLAS/Triton 的 FP32 累加序差异 → hidden state ~1e-6 变化 → 后续 42 层 MoE routing 改变 → File loads ±20%. 不得做 shared expert 的 A/B 对比 — 路由噪声淹没收益.
- 修 bug 先写 L1 测试再修代码

## 使用 MTP

```bash
make profile -- --use-mtp --mtp-eager    # Eager 模式, ~2.68 t/s (2.7×)
make profile -- --use-mtp               # 验证模式, ~0.58 t/s (慢于基线)

# 日常评估
make test-unit                           # 单元测试 (含 13 个 MTP 测试)
```

## 已知陷阱 (gotchas)

- `ExpertWeightCache.clear()` 原实现清空 pinned 条目 → 任何预加载机制必须修复或检查此行为
- 共享 RAID 多线程读盘反效果 (I/O 争用), prefetch 默认禁用
- `tl.load/store` 必须有行列掩码; `tl.dot` 要求 M,N,K≥16
- M=1 decode 时 fused Triton kernel 比 cuBLAS 慢 8× (15/16 SM 空转)
- 修改 Triton kernel 后删 `~/.triton/cache/`
- CPU 全量预载 (11008 专家, 169GB RAM) 反效果 — 挤占 page cache, 推理 +8%
- **Shared expert cuBLAS 修改导致路由噪声**: cuBLAS vs Triton 的 FP32 累加序差异 → 后续 42 层 MoE routing 改变 → File loads ±20%. **不得用于 A/B 对比**.
- **Per-layer hot experts**: 43 层各 48 热专家可能全部不同 (256 个专家全覆盖). GPU hot cache (64 slots) 对跨层覆盖不足, 收益来自 CPU ExpertWeightCache pinning.
- **Prefill/decode 必须分离**: `make profile` 输出 "Decode throughput" (真实生成速度). 旧 "Throughput" 含 prefill, 低估 25-30%.
- **MTP FP8 权重必须 scale-aware 反量化**: `load_fp8_weight` (`cast_back`) vs `.to(bf16)` 丢失 scale → 接受率 0%. 已在 `_load_mtp_weights` 中修复.
- **MTP eager 改变输出分布**: 无验证，draft 直接接受 → token trajectory 偏离基线. 输出语义连贯但与基线不同.
- **MTP 验证慢于基线**: 顺序验证 3 draft × 43 layers = 129 passes, 即使 100% 接受也仅持平. MHC kernel (`mhc_post`) 只支持 T=1, 无法做批处理 forward.
- **MTP 验证接受率 ~2%**: 全 MTP 模块 (attention + MoE FFN + MHC) 下仍低. 跟因可能是 FP8 累加序 / MHC kernel 精度差异导致 MTP 模块分布与主模型不一致.

## Triton Kernel 铁律

- `tl.load` / `tl.store` 必须有行列掩码 (M < BM 时必加行掩码)
- `tl.dot` 最小: M≥16, N≥16, K≥16
- 修改 kernel 后删 `~/.triton/cache/`
- 不得在 `@triton.jit` 内定义嵌套函数
