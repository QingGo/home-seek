# Home-Seek

DeepSeek-V4-Flash 单卡/多卡 RTX 推理引擎。V21.14 — Profiling overhaul + 瓶颈重新定位。

## 纪律

1. 修 bug 必须先写复现该 bug 的快速单元测试 (L1), 再修代码
2. 功能里程碑完成后，或实现某个性能优化后，必须 `make profile`(看全，不要只看其中一部分), 对照 `.agent_memory.md` 基线检查，并确认推理结果无异常。
3. 提交前必须通过 `make lint test-unit`

## 命令

```bash
make install           # 首次或依赖变更后
make lint              # ruff 静态检查
make test-unit         # 单元测试 (304 pass, 2 skip)
make test-integration  # 集成测试 (需 weights/)
make profile           # 标准 profile (5 prompts, 30 tokens). 现在包含: intra-FFN细分, decode step分布, post_step overhead
make profile-compare   # 对比 prev/last profile (现在 per-layer + cache + intra-FFN + BW 全部对比)
make profile-nsys      # Nsight Systems: CUDA stream 时间线 + PCIe 传输 (现在单GPU路径有 NVTX: ffn_routing, decode_embed, ...)
make profile-ncu       # Nsight Compute: 单个 kernel 深度分析
make profile-deep      # nsys + torch.profiler 双重 trace
make server            # 启动 API 服务器
make cli               # 交互式客户端

# 多轮 profiling
uv run python -m home_seek.profiling_runner --rounds 5 \
  --prompts "Hello" "What is AI?" "Write a poem" "How are you?" "Hi" \
  --max-tokens 20 --temperature 0

# Chrome trace (torch.profiler)
uv run python -m home_seek.profiling_runner \
  --rounds 2 --prompts "Hello" "Hello" --max-tokens 5 --temperature 0 \
  --profiler chrome --profiler-warmup 1

# Nsight Systems (需安装 nsight-systems)
nsys profile -o trace -t nvtx,cuda,osrt \
  --show-output true --force-overwrite true \
  uv run python -m home_seek.profiling_runner \
    --rounds 1 --prompt "Hello" --max-tokens 2 --temperature 0

# Nsight Compute (需安装 nsight-compute)
ncu --set full --kernel-name "triton_fused" \
  --launch-count 10 \
  uv run python -m home_seek.profiling_runner \
    --rounds 1 --prompt "Hello" --max-tokens 2 --temperature 0

# 查看 trace
nsys-ui trace.nsys-rep   # Nsight Systems GUI
ncu-ui trace.ncu-rep     # Nsight Compute GUI (web)

# MTP
uv run python -m home_seek.profiling_runner --prompt "Hello" --max-tokens 20 --temperature 0 --use-mtp

# Server + CLI
python -m home_seek server --port 8000
python -m home_seek cli --port 8000

# 下载权重
python -m home_seek download                     # ModelScope
python -m home_seek download --source huggingface # HuggingFace
```

所有命令内部使用 `uv run`.

## 项目结构

```
src/home_seek/                     # 主包
├── inference_engine/              # 推理引擎包 (V21 从单文件拆分)
│   ├── __init__.py                # 兼容旧 import 路径
│   ├── engine.py                  # HomeSeekInferenceEngine + 辅助函数
│   ├── weight_loader.py           # WeightLoader + FP4/FP8 加载函数
│   ├── layer_state.py             # LayerState (per-layer KV 状态)
│   └── expert_cache.py            # ExpertWeightCache + ExpertCacheManager
├── fused_moe.py                   # FusedMoEFFN + SharedExpertFFN (Triton + cuBLAS)
├── router.py                      # MoE 路由 (softplus+sqrt+stable_topk)
├── compressor.py                  # KV 压缩 (支持 T>1 decode)
├── _fp4.py                        # FP4 量化/反量化工具
├── hybrid_kv_cache.py             # Hybrid KV Cache
├── lightning_indexer.py           # Lightning Attention indexer
├── model_config.py                # @dataclass 配置 (V21 从手写类迁移)
├── mhc.py                         # MHC split sinkhorn
├── expert_predictor.py            # 专家预测器 (RecordingPredictor)
├── hw_profile.py                  # 硬件探测
├── profiling_runner.py            # 性能分析入口
├── quantize_weights.py            # 权重量化脚本
└── utils.py                       # rms_norm 工具函数

scripts/                           # 独立工具脚本
├── hot_expert_analyzer.py         # 生成 hot_experts.json
├── hw_probe.py                    # 硬件性能探测
├── analyze_weights.py             # 权重结构分析
├── list_weights.py                # 权重文件列表
├── mem_profiler.py                # 内存分析
├── mem_stress_test.py             # 内存压力测试
└── test_scenarios.py              # 集成测试场景

tests/                             # 测试
├── conftest.py                    # Triton 预热 + CUDA 自动 skip + 确定性种子
├── _reference.py                  # 测试专用参考实现 (swiglu, reduce/expand fused)
├── _engine_stub.py                # EngineStub 统一测试构造 (V21 新增)
├── test_fixes.py
├── test_fp4_experts.py            # 原名 test_v10.py
├── test_gpu_expert_store.py
├── test_mtp.py
├── test_quantization.py
├── test_tile_ops.py
├── test_weight_loading_bug.py     # +契约测试 (TestExpertCacheContract)
└── integration/
    ├── conftest.py
    └── test_inference_e2e.py      # +回归哈希测试
```

## 关键路径

- 模型: `weights/` (46 safetensors, ~150GB); 论文: `docs/paper.md`
- 热专家: `hot_experts.json`; 架构设计: `docs/arch_design.md`
- 实施记录: `docs/implementation_notes.md`
- 里程碑记忆: `.agent_memory.md` (基线+瓶颈+下一步)

## 硬件配置 (HardwareConfig)

所有硬件参数集中管理在 `HardwareConfig` (src/home_seek/hardware_config.py)：
- `HardwareConfig.auto(hw, cfg, **overrides)` 从 `HWProfile` 自动推导
- 预设策略按 GPU 名子串匹配 (`"4090" in gpu_name.lower()`)
- 不在列表的 GPU 走 fallback + VRAM 公式自适应
- **多 GPU 自动探测**: `hw.n_gpu > 1` 且策略未设 `devices` → 自动生成 device 列表 + device_map
- 策略新增 `per_gpu_*` 语义字段控制单卡 cap，`_compute` 根据 n_gpu 自动扩展
- 启动时日志示例:
  ```
  Matched strategy: 4090
  Auto multi-GPU: 8 × NVIDIA GeForce RTX 4090
  HardwareConfig: RTX 4090  ×8  VRAM=24GB  SM=128  hot=64  ...  devices=8  device_map=43layers
  Multi-GPU device map (43 layers): GPU0:6layers, ..., GPU7:1layers
  ```

## 并行后端 (ParallelBackend)

所有并行策略通过 `src/home_seek/inference_engine/parallel.py` 管理：
- `ParallelBackend` — 抽象基类, 定义 `layer_device`, `transfer_hidden`, `resolve_expert_device`
- `PPBackend` — Pipeline Parallel (当前实现): 层均分, h 串行传递, per-GPU 状态
- `EPBackend` — Expert Parallel (预留): all-to-all 路由
- `TPBackend` — Tensor Parallel (预留): all-reduce 聚合
- `_ensure_backend()` 惰性创建, `__new__` 测试桩自动兼容
- per-GPU 状态: `LayerState`, `Compressor`, `Indexer`, `HybridKVCache`, expert caches
- embed/lm_head/norm_weight 自动复制到所有 GPU

## 缓存体系 (V21.4)

```
请求 expert (layer, eid)
  └─ ExpertCacheManager (统一入口, 但引擎实际走自己的 _gpu_hot / _gpu_bf16 / ExpertWeightCache)
       ├─ 1. _gpu_hot (GPU BF16, ~64 × 48MB, FIFO evict; 启动时预装 hot_experts.json)
       ├─ 2. _gpu_bf16 (GPU BF16 LRU, ~100 × 48MB, 自动淘汰)
       ├─ 3. ExpertWeightCache (CPU FP4, ~3292 条, pin=永不淘汰)
       │    ├─ ~2103 pinned (hot×43 + hash×3 + MTP×256)
       │    └─ ~1189 unpinned (LRU, 跨层条带预载)
       └─ 4. safetensors mmap (page cache, RAID 1.5 GB/s)

V21.4 关键: _per_expert_bytes = I*D*51//32 ≈ 12.75 MB (f8 scale 不转 fp32).
cache 大小与 page cache 平衡: ~45 GB CPU cache + ~45 GB page cache. 留一半 RAM 给 OS.

共享专家:
  └─ _shared_expert_weights (GPU, 43 层 FP8, lazily dequant → BF16)
     └─ M=1 decode: cuBLAS, Triton if M>1

MTP 模块 (V21.4):
  ├─ _mtp_weights: 33 非专家权重 (GPU BF16)
  ├─ ExpertWeightCache: 256 专家 (CPU FP4 pinned)
  ├─ _mtp_generate_draft: 自回归生成 draft (默认 M=2, argmax 当 temperature=0)
  └─ _mtp_verify_batched: 批验证 (43 层, causal mask, temperature 传播)

_forward_layer 共享方法: 所有层 forward 走 self._forward_layer(h, lw, layer_idx, input_ids).
```

Server: `home-seek server` / `home-seek cli` / `home-seek download`
- Server 使用线程级 HTTP server (std lib), 避免 uvicorn fork+CUDA 不兼容
- CLI 通过 SSE 流式输出, ANSI 颜色自动禁用(管道/重定向)
- 每轮独立统计: prefill t/s, decode t/s, TTFT

## 性能评估铁律

- **`--temperature 0` (argmax)** 做 A/B 对比 (路由噪声 ±15%)
- **3+ 次重复取平均**: 磁盘负载波动 ±2-3%
- **指标**: Decode throughput (t/s) + File loads
- **Prefill/decode 分离**: `decode_time_s` / `num_generated_tokens`
- **Shared expert 修改**: cuBLAS vs Triton 的 FP32 累加序差异 → 路由噪声 ±20%. **不得用于 A/B 对比**
- **Profile 自动保存**: `make profile` 输出 JSON 到 `artifacts/last_profile.json`, `make profile-compare` 对比前后变化
- **Profiling tax**: 每个 trace wrapper 前的 `torch.cuda.synchronize()` 增加 ~100-200ms/tok 开销. 生产环境无 tracing 时 decode 快 ~10-15%. Bottleneck analysis 的 "Other" decomposition 会分解此开销.
- 修 bug 先写 L1 测试

## 已知陷阱 (gotchas)

- **模型真正的结束标记是 token 1 (`</｜end▁of▁sentence｜>`)**, 不是 EOS 128000. `_load_stop_token_ids()` 从 `weights/tokenizer.json` 读取. 找不到文件直接报错.
- **FusedMoEFFN cuBLAS 小 M**: `fused_expert_ffn_triton` 在 M<=8 时走 cuBLAS, 避免 Triton 15/16 SM 空转. `fused_moe.py:297`.
- `ExpertWeightCache.clear()` 保留 pinned 条目; `ExpertCacheManager.clear()` 清空全部
- 共享 RAID 多线程读盘反效果, prefetch 默认禁用. V21 已移除 prefetch 代码 (移入 scripts/)
- `tl.load/store` 必须有行列掩码; `tl.dot` M,N,K≥16
- M=1 decode: Triton 比 cuBLAS 慢 8× (15/16 SM 空转). SharedExpertFFN 和 FusedMoEFFN 都应考虑此限制
- 修改 Triton kernel 后删 `~/.triton/cache/`
- CPU 全量预载 (11008 专家, 169GB) 挤占 page cache, 推理 +8%
- **Per-layer hot experts**: 43 层各 48 热可能全不同, GPU hot cache 覆盖不足
- **MTP FP8 必须 scale-aware**: `load_fp8_weight` (cast_back) vs `.to(bf16)` 丢失 scale → 接受率 0%
- **MTP eager 输出漂移**: 无验证直接接受 draft → trajectory 偏离基线
- **MTP 验证状态回滚必须完整**: `compressed_kv_data`, `compressed_kv_idx` 也需保存/恢复, 仅 `kv_latent_cache` 不够
- **MTP 批验证 causal mask**: `_forward_attn` 在 T>1 时添加 triu mask 在 k_sw 末尾 T 个位置. 无 T=1 开销
- **KV session 文件**: `sessions/{session_id}/` 存所有层状态, 恢复时自动 map_location 到 GPU
- **温度 0 对比必须**: 消除路由噪声
- **MHC_post Triton kernel 始终失败** (AssertionError): 直走 PyTorch fallback
- **EngineStub 是推荐的测试构造方式**: 用 `from tests._engine_stub import make_engine` 替代 `HomeSeekInferenceEngine.__new__()` + 手工属性赋值
- **Config 现在是 dataclass**: `DeepSeekV4FlashConfig(head_dim=512)` 而非 `type('obj', ...)()`; `from_json()` 自动通过 `__dataclass_fields__` 映射字段
- **`_forward_layer` 共享方法**: 不要直接复制粘贴层循环; 所有层 forward 都走 `self._forward_layer(h, lw, layer_idx, input_ids)`
- **`encoding_dsv4` 路径**: V21 使用 `_project_root` 绝对路径计算, 不依赖 `__file__` 相对层级
- **日志**: 使用 `logging.getLogger(__name__)` 而非 `print()`. `_log` 方法内部调用 `_logger.info()`
- **V21.4 Thread-safe cache**: `ExpertWeightCache.put()` 包装 KeyError 处理多线程并发 eviction
- **GPU store 命中率可能为 0%**: `hot_experts.json` 与真实路由可能完全不重合. 检查 `make profile` 输出中的 `GPU store: 0h/0m`. 如果为 0, 所有 expert 都走 CPU→GPU DMA, 重新生成 hot_experts.json 可能改善.
- **Bottleneck analysis 数字使用最后一段的 delta**: 多轮 profile 时 bottleneck 分析使用 `round_results[-1]` 的 per-round delta, 不是 `layer_trace` (累积所有轮).
- **post_step = ~22ms 是真实值**: lm_head matmul 不是瓶颈. 不要花时间优化它.

## Triton Kernel 铁律

- `tl.load` / `tl.store` 必须有行列掩码 (M < BM 时必加行掩码)
- `tl.dot` 最小: M≥16, N≥16, K≥16
- 修改 kernel 后删 `~/.triton/cache/`
- 不得在 `@triton.jit` 内定义嵌套函数
