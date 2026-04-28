# Home-Seek

DeepSeek-V4-Flash 推理引擎 — 单卡 RTX 4090 运行 284B MoE 模型。

## 安装

```bash
pip install home-seek

# 下载模型权重 (约 150GB)
huggingface-cli download QingGo/Home-Seek --local-dir weights
```

或从源码安装:

```bash
git clone https://github.com/QingGo/home-seek.git
cd home-seek
pip install -e .
```

## 快速开始

```bash
# 启动 API 服务器
home-seek server --port 8000

# 另一个终端 — 交互式 CLI
home-seek cli --port 8000
```

CLI 支持的命令:
- `/think` — 切换思考/非思考模式
- `/stats` — 显示上一轮统计数据
- `/help` — 帮助
- `/quit` — 退出

## API

OpenAI 兼容接口:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50,
    "temperature": 0
  }'
```

### 思考模式

```bash
curl ... -d '{
  "messages": [...],
  "thinking": true
}'
```

### 流式输出

```bash
curl ... -d '{
  "messages": [...],
  "stream": true
}'
```

每个响应包含 `stats` 字段:

```json
{
  "stats": {
    "encoding_tokens": 5,
    "encoding_time_ms": 7952,
    "encoding_speed_tps": 0.6,
    "ttft_ms": 7952,
    "generated_tokens": 9,
    "decode_time_ms": 6255,
    "decode_speed_tps": 0.8
  }
}
```

## 性能分析

```bash
make profile       # 单轮 profiling
python -m home_seek.profiling_runner --rounds 5 \
  --prompts "Hello" "What is AI?" "Write a poem" \
  --max-tokens 20 --temperature 0
```

## 系统要求

- NVIDIA RTX 4090 (24GB VRAM)
- 90GB+ 系统内存 (容器/VM)
- 150GB 磁盘空间 (模型权重)
- Linux + CUDA 12+

## 架构

- FP4 量化专家权重, CPU pinned memory 常驻
- MoE 路由专家 (256/层) + 共享专家
- Triton 反量化 + cuBLAS matmul
- 三层 KV 缓存 (SWA + CSA + HCA)
- MHC (Multi-Head Concat) 残差混合
