import os
import json
import torch
from collections import Counter, defaultdict
from home_seek.inference_engine import HomeSeekInferenceEngine, rms_norm
from transformers import PreTrainedTokenizerFast


def generate_calibration_corpus(tokenizer, num_tokens: int = 8192, seed: int = 42):
    torch.manual_seed(seed)

    def rep(text, n):
        return text * n

    categories = [
        # Natural language — diverse topics (each ~150 tokens, repeat 6x = ~2600)
        rep("The quick brown fox jumps over the lazy dog. "
            "The five boxing wizards jump quickly. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump. ", 60),

        rep("Machine learning is a subset of artificial intelligence that focuses on the development "
            "of algorithms that allow computers to learn from and make predictions based on data. "
            "Deep learning is a further subset using neural networks with many layers. "
            "Reinforcement learning trains agents through trial and error. "
            "Supervised learning uses labeled data while unsupervised learning finds patterns. ", 8),

        rep("The capital of France is Paris, a city known for its art, fashion, and culture. "
            "The Eiffel Tower is one of the most famous landmarks in the world. "
            "Japan has a rich cultural heritage with traditions dating back centuries. "
            "The Great Wall of China stretches over 13000 miles across northern China. "
            "Brazil is the largest country in South America with diverse ecosystems. "
            "Ancient Egypt built pyramids that still stand today as marvels of engineering. ", 6),

        rep("In economics, supply and demand determines price in a market economy. "
            "When demand increases and supply stays constant, prices rise. "
            "Inflation occurs when the general price level rises over time. "
            "Central banks use interest rates to control inflation and stimulate growth. "
            "GDP measures the total value of goods and services produced in a country. "
            "Fiscal policy involves government spending and taxation decisions. ", 8),

        rep("The human brain contains approximately 86 billion neurons forming complex networks. "
            "Neuroplasticity allows the brain to reorganize itself throughout life. "
            "DNA stores genetic information in a double helix structure discovered by Watson and Crick. "
            "Mitochondria are the powerhouses of the cell, generating energy through ATP production. "
            "The theory of evolution by natural selection was proposed by Charles Darwin. "
            "Photosynthesis converts light energy into chemical energy in plants. ", 8),

        # Code — algorithms (each ~100 tokens, repeat 6x = ~2800)
        rep("def quicksort(arr):\n"
            "    if len(arr) <= 1: return arr\n"
            "    pivot = arr[len(arr) // 2]\n"
            "    left = [x for x in arr if x < pivot]\n"
            "    middle = [x for x in arr if x == pivot]\n"
            "    right = [x for x in arr if x > pivot]\n"
            "    return quicksort(left) + middle + quicksort(right)\n\n"
            "def binary_search(arr, target):\n"
            "    low, high = 0, len(arr) - 1\n"
            "    while low <= high:\n"
            "        mid = (low + high) // 2\n"
            "        if arr[mid] == target: return mid\n"
            "        elif arr[mid] < target: low = mid + 1\n"
            "        else: high = mid - 1\n"
            "    return -1\n\n", 6),

        rep("import torch\nimport torch.nn as nn\n\n"
            "class TransformerBlock(nn.Module):\n"
            "    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):\n"
            "        super().__init__()\n"
            "        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)\n"
            "        self.ffn = nn.Sequential(\n"
            "            nn.Linear(d_model, dim_feedforward),\n"
            "            nn.ReLU(),\n"
            "            nn.Linear(dim_feedforward, d_model),\n"
            "        )\n"
            "        self.norm1 = nn.LayerNorm(d_model)\n"
            "        self.norm2 = nn.LayerNorm(d_model)\n"
            "        self.dropout = nn.Dropout(dropout)\n\n"
            "    def forward(self, x):\n"
            "        x2 = self.norm1(x)\n"
            "        x = x + self.dropout(self.self_attn(x2, x2, x2)[0])\n"
            "        x2 = self.norm2(x)\n"
            "        x = x + self.dropout(self.ffn(x2))\n"
            "        return x\n\n"
            "class ResidualBlock(nn.Module):\n"
            "    def __init__(self, dim):\n"
            "        super().__init__()\n"
            "        self.linear1 = nn.Linear(dim, dim)\n"
            "        self.linear2 = nn.Linear(dim, dim)\n"
            "        self.norm = nn.LayerNorm(dim)\n\n"
            "    def forward(self, x):\n"
            "        return x + self.linear2(torch.relu(self.linear1(self.norm(x))))\n", 4),

        rep("import asyncio\nimport aiohttp\n\n"
            "async def fetch_url(session, url):\n"
            "    async with session.get(url) as response:\n"
            "        return await response.text()\n\n"
            "async def fetch_all(urls):\n"
            "    async with aiohttp.ClientSession() as session:\n"
            "        tasks = [fetch_url(session, url) for url in urls]\n"
            "        return await asyncio.gather(*tasks)\n\n"
            "async def main():\n"
            "    data = await fetch_all(['https://api.example.com/data'] * 10)\n"
            "    results = [json.loads(d) for d in data]\n"
            "    return results\n", 6),

        # Math — equations and derivations
        rep("The quadratic formula: x = (-b ± sqrt(b² - 4ac)) / (2a). "
            "The derivative of x² is 2x. The integral of eˣ dx is eˣ + C. "
            "The Taylor series expands eˣ as sum_{n=0}^{∞} x^n / n!. "
            "The Fourier transform converts time domain to frequency domain. "
            "Gradient descent updates: θ = θ - α ∇J(θ). "
            "Softmax normalizes: softmax(x)_i = exp(x_i) / sum_j exp(x_j). "
            "Attention mechanism: Attention(Q,K,V) = softmax(QK^T / sqrt(d)) V. "
            "The chain rule: dz/dx = dz/dy * dy/dx. "
            "Bayes theorem: P(A|B) = P(B|A) P(A) / P(B). "
            "Cross-entropy loss: H(p,q) = -∑ p(x) log q(x). ", 8),

        # SQL and databases
        rep("SELECT u.name, COUNT(o.id) as order_count\n"
            "FROM users u\n"
            "LEFT JOIN orders o ON u.id = o.user_id\n"
            "WHERE u.created_at > '2024-01-01'\n"
            "GROUP BY u.id, u.name\n"
            "HAVING COUNT(o.id) > 5\n"
            "ORDER BY order_count DESC\n"
            "LIMIT 100;\n\n"
            "SELECT p.name, AVG(r.rating) as avg_rating\n"
            "FROM products p\n"
            "JOIN reviews r ON p.id = r.product_id\n"
            "WHERE r.created_at > '2024-06-01'\n"
            "GROUP BY p.id, p.name\n"
            "HAVING AVG(r.rating) >= 4.0\n"
            "ORDER BY avg_rating DESC;\n", 8),

        # Scientific computing
        rep("import numpy as np\nfrom scipy import optimize, linalg\n\n"
            "def rosenbrock(x):\n"
            "    return sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2)\n\n"
            "result = optimize.minimize(rosenbrock, np.zeros(10), method='L-BFGS-B')\n"
            "A = np.random.randn(100, 50)\n"
            "b = np.random.randn(100)\n"
            "x_hat, _, _, _ = linalg.lstsq(A, b)\n"
            "residuals = A @ x_hat - b\n"
            "mse = np.mean(residuals**2)\n", 8),

        # Structured data
        rep('{"name": "Alice", "age": 30, "city": "New York", "score": 95.5}\n'
            '{"name": "Bob", "age": 25, "city": "San Francisco", "score": 87.2}\n'
            '{"name": "Charlie", "age": 35, "city": "London", "score": 91.8}\n'
            '{"name": "Diana", "age": 28, "city": "Tokyo", "score": 88.4}\n'
            '{"name": "Eve", "age": 32, "city": "Berlin", "score": 93.1}\n', 8),

        # Systems programming
        rep("int main(int argc, char *argv[]) {\n"
            "    int fd = open(\"data.txt\", O_RDONLY);\n"
            "    if (fd < 0) { perror(\"open\"); return 1; }\n"
            "    char buf[4096];\n"
            "    ssize_t n = read(fd, buf, sizeof(buf));\n"
            "    if (n < 0) { perror(\"read\"); close(fd); return 1; }\n"
            "    printf(\"Read %zd bytes\\n\", n);\n"
            "    close(fd);\n"
            "    return 0;\n"
            "}\n", 6),
    ]
    ids_list = []
    total = 0
    for cat in categories:
        ids = tokenizer.encode(cat)
        ids_list.extend(ids)
        total += len(ids)
        if total >= num_tokens:
            break
    return torch.tensor(ids_list[:num_tokens], dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def analyze_hot_experts(weight_dir: str, num_tokens: int = 8192, num_hot: int = 48,
                        output_path: str = "hot_experts.json"):
    print(f"[hot_expert] Loading model (calibration: {num_tokens} tok, top-{num_hot})...")
    eng = HomeSeekInferenceEngine(weight_dir, verbose=False)
    tokenizer_path = os.path.join(weight_dir, "tokenizer.json")
    tok = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    print("[hot_expert] Generating calibration corpus...")
    all_input_ids = generate_calibration_corpus(tok, num_tokens=num_tokens)
    actual_tokens = min(num_tokens, all_input_ids.shape[1])
    all_input_ids = all_input_ids[:, :actual_tokens]

    chunk_size = 512
    hc_mult = getattr(eng.config, 'hc_mult', 1)

    expert_counts = Counter()
    layer_expert_trace = defaultdict(list)

    for chunk_start in range(0, actual_tokens, chunk_size):
        chunk_end = min(chunk_start + chunk_size, actual_tokens)
        input_ids = all_input_ids[:, chunk_start:chunk_end].to(eng.device)
        eng.layer_states = {}

        h = eng.embed[input_ids].to(torch.bfloat16)
        h = h.unsqueeze(2).expand(-1, -1, hc_mult, -1)

        for layer_idx in range(eng.config.num_hidden_layers):
            lw = eng._get_layer_weights(layer_idx)

            # === Attention block ===
            residual_attn = h
            h_pre_attn, post, comb = eng._process_mhc_layer(h, lw, "hc_attn")
            if lw.get("attn_norm.weight") is not None:
                h_norm = rms_norm(h_pre_attn, lw["attn_norm.weight"].to(torch.bfloat16),
                                  eng.config.rms_norm_eps)
            else:
                h_norm = h_pre_attn
            if h_norm.shape[1] > 0:
                attn_out = eng._forward_attn(h_norm, lw, layer_idx)
            else:
                attn_out = torch.zeros_like(h_norm)
            if post is not None and comb is not None:
                h = eng._process_mhc_post(attn_out, residual_attn, post, comb)
            else:
                h = h + attn_out.unsqueeze(2).expand(-1, -1, hc_mult, -1)

            # === FFN block ===
            residual_ffn = h
            h_pre_ffn, ffn_post, ffn_comb = eng._process_mhc_layer(h, lw, "hc_ffn")
            if lw.get("ffn_norm.weight") is not None:
                h_ffn_norm = rms_norm(h_pre_ffn, lw["ffn_norm.weight"].to(torch.bfloat16),
                                      eng.config.rms_norm_eps)
            else:
                h_ffn_norm = h_pre_ffn

            gate_w = eng._deq("ffn.gate", lw.get("ffn.gate.weight"), lw.get("ffn.gate.scale"))
            gate_bias = lw.get("ffn.gate.bias")

            if layer_idx < eng.config.num_hash_layers:
                tid2eid = lw.get("ffn.gate.tid2eid")
                if tid2eid is not None:
                    eids = tid2eid[input_ids]
                    for b in range(eids.shape[0]):
                        for t in range(eids.shape[1]):
                            for k in range(eids.shape[2]):
                                eid = eids[b, t, k].item()
                                expert_counts[eid] += 1
                                layer_expert_trace[layer_idx].append(eid)

            if gate_w is not None:
                scores = torch.matmul(h_ffn_norm.to(gate_w.dtype), gate_w.t())
                if gate_bias is not None:
                    scores = scores + gate_bias.to(scores.dtype)
                scores = torch.nn.functional.softplus(scores).sqrt()
                _, topk_idx = torch.topk(scores, eng.config.num_experts_per_tok, dim=-1)
                for b in range(topk_idx.shape[0]):
                    for t in range(topk_idx.shape[1]):
                        for k in range(topk_idx.shape[2]):
                            eid = topk_idx[b, t, k].item()
                            expert_counts[eid] += 1
                            layer_expert_trace[layer_idx].append(eid)

            ffn_out, _ = eng._forward_ffn(h_ffn_norm, lw, layer_idx, input_ids)
            if ffn_post is not None and ffn_comb is not None:
                h = eng._process_mhc_post(ffn_out, residual_ffn, ffn_post, ffn_comb)
            else:
                h = h + ffn_out.unsqueeze(2).expand(-1, -1, hc_mult, -1)

        # Clean up per-chunk state
        eng.layer_states = {}
        torch.cuda.empty_cache()

        if (chunk_start // chunk_size + 1) % 4 == 0:
            print(f"[hot_expert]  Chunk {chunk_start//chunk_size+1}/{(actual_tokens+chunk_size-1)//chunk_size} "
                  f"(token {chunk_start}/{actual_tokens}) done")

    total_calls = sum(expert_counts.values())
    top_hot = [eid for eid, _ in expert_counts.most_common(num_hot)]
    top_hot_coverage = sum(expert_counts[eid] for eid in top_hot) / total_calls if total_calls > 0 else 0
    top_16 = top_hot[:16]
    top_16_coverage = sum(expert_counts[eid] for eid in top_16) / total_calls if total_calls > 0 else 0

    result = {
        "top_hot_experts": top_hot,
        "top_hot_coverage": round(top_hot_coverage, 4),
        "num_hot_experts": num_hot,
        "top_16_hot_experts": top_16,
        "top_16_coverage": round(top_16_coverage, 4),
        "total_expert_calls": total_calls,
        "calibration_tokens": actual_tokens,
        "all_expert_counts": [(int(eid), int(cnt)) for eid, cnt in expert_counts.most_common(50)],
        "hash_layer_expert_ids": list(range(18)),
        "config": {
            "num_layers": eng.config.num_hidden_layers,
            "num_experts": eng.config.n_routed_experts,
            "num_hash_layers": eng.config.num_hash_layers,
        },
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[hot_expert] Top-{num_hot} experts: {top_hot}")
    print(f"[hot_expert] Top-{num_hot} coverage: {top_hot_coverage:.2%}")
    print(f"[hot_expert] Top-16 coverage (legacy): {top_16_coverage:.2%}")
    print(f"[hot_expert] Saved to {output_path}")
    return result


if __name__ == "__main__":
    import sys
    weight_dir = sys.argv[1] if len(sys.argv) > 1 else "weights"
    output = sys.argv[2] if len(sys.argv) > 2 else "hot_experts.json"
    analyze_hot_experts(weight_dir, output_path=output)
