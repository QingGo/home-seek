# Home-Seek 开发指南

DeepSeek-V4-Flash 推理引擎。架构背景见 `docs/arch_design.md`。
长期记忆 (里程碑、决策、踩坑) 见 `.agent_memory.md`。

## 命令

```bash
# 依赖
uv sync

# 静态检查
ruff check home_seek/ tests/

# 快速单测 (~10s, 无权重)
.venv/bin/python -m pytest tests/ --ignore=tests/integration -m "not slow" -q

# 含集成测试 (~100s, 需 weights/)
.venv/bin/python -m pytest tests/integration/ -q

# 入口
.venv/bin/python main.py --prompt "Hello" --max-tokens 8 --verbose

# Profiling (每次大功能完成后必须执行)
.venv/bin/python -m home_seek.profiling_runner --prompt "Hello" --max-tokens 8
```

## Profiling 检查清单

每次大功能完成后执行 profiling，对照以下维度确认：

| 检查项 | 判断标准 |
|--------|---------|
| 输出是否正确 | 生成文本语义通顺、无明显乱码 |
| 吞吐提升 | 对照 `.agent_memory.md` 当前基线是否有预期改善 |
| 瓶颈是否转移 | FFN 占比是否下降、新模块是否异常 |
| 无性能回归 | 各指标 (延迟、显存、I/O) 不应变差 |

## 测试基础设施 (`tests/conftest.py`)

- **Session 级 Triton 预热** — 所有 kernel 在 session 启动时编译一次, 消除每文件首次 JIT 延迟.
- **确定性种子** — 每测试基于 `nodeid` MD5 派生种子, 同一测试永远同一种子. 测试内可调用 `torch.manual_seed()` 显式覆盖.
- **显存自动清理** — `autouse` fixture, 每测试前后 `empty_cache()` + `reset_peak_memory_stats()`.
- **权重守卫** — `require_weights` fixture 检查 `weights/model.safetensors.index.json`, 不存在则 skip. 依赖权重的测试类加 `@pytest.mark.usefixtures("require_weights")`.

## Triton Kernel 强制规则

1. **不得使用嵌套函数** — `def` inside `@triton.jit` 编译报错.
2. **不得调用模块级 Python 函数** — 解码逻辑必须内联到 kernel body.
3. **constexpr 参数用短名统一** (`BM, BN, BK`), 调用时用同名 keyword.
4. **每个 `tl.load` 必须有行列掩码** — `M < BM` 时 (decode 阶段 M=1, BM=16) 缺行掩码 = 越界:
   ```python
   mask_m = offs_m < M
   mask_k = (k + offs_k) < K
   x = tl.load(ptr, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
   ```
5. **每个 `tl.store` 必须有行列掩码**:
   ```python
   tl.store(ptr, val, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
   ```

`tl.dot` 最小: M≥16, N≥16, K≥16. RTX 4090 共享内存 ~100KB, 大 tile 需 `num_stages=1`.

## 权重格式陷阱

| 陷阱 | 正解 |
|------|------|
| 路由专家 w2 是 FP8? | **FP4** (int8 packed), 与 w1/w3 相同格式 |
| SwiGLU = sigmoid(gate) × up? | **SiLU(gate) × up = gate × sigmoid(gate) × up** |
| FP4 scale 是 UE8M0 (uint8→int32→bitcast)? | **float8_e8m0fnu**, `.to(torch.float32)` 即得正确值 |
| safetensors 加载的 tensor 已在 GPU? | **在 CPU**, 需显式 `.to(device)` |

## 常见越界与缓存陷阱

| 陷阱 | 正解 |
|------|------|
| `str(device)` 比较 | 用 `device.type != other.type`, `"cuda"` ≠ `"cuda:0"` 会导致不必要的 tensor copy，破坏 `data_ptr` 缓存 |
| `_forward_legacy` (total_slots ≤ 64) | decode 阶段 (1 token × 6 experts = 6) 永远走此路径，必须支持 6-tuple |
| 反量化缓存 | 模块级 `_deq_cache` (64 条目, key=`w1_packed.data_ptr()`)，每次 `generate()` 前 `clear_deq_cache()` 清空 |

## 调试

- `CUDA_LAUNCH_BLOCKING=1` 定位 CUDA 越界的真实位置 (否则错误异步报告, 堆栈错位).
- 修改 Triton kernel 后删 `~/.triton/cache/` 清除 JIT 缓存, 否则加载旧编译版本.
- `HomeSeekInferenceEngine._warmup()` 返回 `bool`. 失败时 `_warmed_up = False`, 首次推理在关键路径上 JIT 编译 (慢但不报错).

## 关键架构

- **GPU FP4 Store**: 512 条目 LRU, 键为 `(layer_idx, expert_idx)`. 无独立 BF16 热缓存.
- **MoE FFN**: FP4 先经 `_dequantize_fp4_to_bf16` 反量化, 再走 2 个 BF16 Triton kernel (`gate_up` + `down`). 模块级 `_deq_cache` (64 条目, 按 `data_ptr` 索引) 避免重复反量化.
- **MoE FFN**: FP4 先经 `_dequantize_fp4_to_bf16` 反量化, 再走 2 个 BF16 Triton kernel (`gate_up` + `down`). 模块级 `_deq_cache` (64 条目, 按 `data_ptr` 索引) 避免重复反量化.
- **预取**: `AsyncPrefetchWorker` 在 `__init__` 时初始化, 基于当前层 hidden state 预测下一层专家.
- **权重加载**: `WeightLoader` 用 mmap 缓存在 `safetensors` 上, 跨层共享. `generate()` 结束时 `loader.close()`.
