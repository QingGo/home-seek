# Home-Seek

DeepSeek-V4-Flash 单卡 RTX 4090 推理引擎。V18 — 缓存重构 + MTP 批验证 + KV 持久化。

## 纪律

1. 修 bug 必须先写复现该 bug 的快速单元测试 (L1), 再修代码
2. 功能里程碑完成后，或实现某个性能优化后，必须 `make profile`, 对照 `.agent_memory.md` 基线检查，并确认推理结果无异常。
3. 提交前必须通过 `make lint test-unit`

## 命令

```bash
make install           # 首次或依赖变更后
make lint              # ruff 静态检查
make test-unit         # 单元测试 (174 pass, <15s)
make test-integration  # 集成测试 (需 weights/)
make profile           # 性能分析: 5+20 tok, temp=0
make smoke             # 最小冒烟

# MTP
make profile -- --use-mtp --mtp-eager   # Eager, ~2.7 t/s
make profile -- --use-mtp               # 验证, ~0.73 t/s
make test-unit                          # 含 13 个 MTP 测试
```

所有命令内部使用 `uv run`.

## 关键路径

- 模型: `weights/` (46 safetensors, ~150GB); 论文: `docs/paper.md`
- 热专家: `hot_experts.json`; 架构设计: `docs/arch_design.md`
- 实施记录: `docs/implementation_notes.md`
- 里程碑记忆: `.agent_memory.md` (基线+瓶颈+下一步)
- 主引擎: `home_seek/inference_engine.py` (~2700 行)
- MoE FFN: `home_seek/fused_moe.py` (Triton FP4 dequant + cuBLAS)
- KV 压缩: `home_seek/compressor.py` (支持 T>1 decode)
- 路由: `home_seek/router.py` (softplus+sqrt+stable_topk)
- 配置: `home_seek/model_config.py`

## 缓存体系 (V18)

```
请求 expert (layer, eid)
  ├─ 1. _gpu_hot_experts (GPU BF16, ~64 × 48MB, per-layer LRU, pinned)
  ├─ 2. _gpu_bf16_cache  (GPU BF16 LRU, ~76 × 48MB, 自动淘汰)
  ├─ 3. ExpertWeightCache (CPU FP4, 5120 条, pin=永不淘汰)
  │    ├─ ~2118 pinned (hot×43 + hash×3)
  │    ├─ ~256 pinned (MTP experts)
  │    └─ ~2746 unpinned (LRU)
  └─ 4. safetensors mmap (RAID 1.5 GB/s)

[V18 移除] _gpu_expert_store (GPU FP4, 命中率 ~0%, 冗余)

共享专家:
  └─ _shared_expert_weights (GPU, 43 层 FP8, lazily dequant → BF16)

MTP 模块:
  ├─ _mtp_weights: 33 非专家权重 (GPU BF16)
  ├─ ExpertWeightCache: 256 专家 (CPU FP4 pinned)
  ├─ _mtp_generate_draft: 自回归生成 draft
  └─ _mtp_verify_batched: 批验证 (T>1 MHC PyTorch fallback)
```

## 性能评估铁律

- **`--temperature 0` (argmax)** 做 A/B 对比 (路由噪声 ±15%)
- **3+ 次重复取平均**: 磁盘负载波动 ±2-3%
- **指标**: Decode throughput (t/s) + File loads
- **Prefill/decode 分离**: `decode_time_s` / `num_generated_tokens`
- **Shared expert 修改**: cuBLAS vs Triton 的 FP32 累加序差异 → 路由噪声 ±20%. **不得用于 A/B 对比**
- **MTP 精度裂谷**: MHC_post Triton (T=1) vs PyTorch (T>1) → hidden ~1e-6 差异 → 43 层放大 → top-1 翻转. 这是接受率 ~2% 的根因
- 修 bug 先写 L1 测试

## 已知陷阱 (gotchas)

- `ExpertWeightCache.clear()` 保留 pinned 条目 (V17 修复)
- 共享 RAID 多线程读盘反效果, prefetch 默认禁用
- `tl.load/store` 必须有行列掩码; `tl.dot` M,N,K≥16
- M=1 decode: Triton 比 cuBLAS 慢 8× (15/16 SM 空转)
- 修改 Triton kernel 后删 `~/.triton/cache/`
- CPU 全量预载 (11008 专家, 169GB) 挤占 page cache, 推理 +8%
- **Per-layer hot experts**: 43 层各 48 热可能全不同, GPU hot cache 覆盖不足
- **MTP FP8 必须 scale-aware**: `load_fp8_weight` (cast_back) vs `.to(bf16)` 丢失 scale → 接受率 0%
- **MTP eager 输出漂移**: 无验证直接接受 draft → trajectory 偏离基线
- **MTP 验证接受率 ~2%**: MHC_post Triton vs PyTorch 精度差异 → MoE 路由偏移 (详见 `.agent_memory.md`)
- **KV session 文件**: `sessions/{session_id}/` 存所有层状态, 恢复时自动 map_location 到 GPU
- **温度 0 对比必须**: 消除路由噪声

## Triton Kernel 铁律

- `tl.load` / `tl.store` 必须有行列掩码 (M < BM 时必加行掩码)
- `tl.dot` 最小: M≥16, N≥16, K≥16
- 修改 kernel 后删 `~/.triton/cache/`
- 不得在 `@triton.jit` 内定义嵌套函数
