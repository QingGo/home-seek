import os
import json
import torch
from collections import Counter, defaultdict
from home_seek.inference_engine import HomeSeekInferenceEngine, rms_norm
from transformers import PreTrainedTokenizerFast


def generate_calibration_corpus(tokenizer, num_tokens: int = 1024, seed: int = 42):
    torch.manual_seed(seed)
    texts = []
    categories = [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)\n" * 5,
        "The integral of x^2 from 0 to 1 is 1/3. " * 10,
        "The capital of France is Paris. " * 15,
        "for i in range(10):\n    print(f'Hello {i}')\n" * 5,
        "Machine learning is a subset of artificial intelligence. " * 10,
        "SELECT * FROM users WHERE age > 18 ORDER BY name;\n" * 5,
        "In this paper, we propose a novel architecture for large language models. " * 10,
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
def analyze_hot_experts(weight_dir: str, num_tokens: int = 2048, output_path: str = "hot_experts.json"):
    print("[hot_expert] Loading model...")
    eng = HomeSeekInferenceEngine(weight_dir, verbose=False)
    tokenizer_path = os.path.join(weight_dir, "tokenizer.json")
    tok = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    print("[hot_expert] Generating calibration corpus...")
    input_ids = generate_calibration_corpus(tok, num_tokens=num_tokens).to(eng.device)
    actual_tokens = min(num_tokens, input_ids.shape[1])
    input_ids = input_ids[:, :actual_tokens]

    eng.layer_states = {}
    h = eng.embed[input_ids].to(torch.bfloat16)

    expert_counts = Counter()
    layer_expert_trace = defaultdict(list)

    for layer_idx in range(eng.config.num_hidden_layers):
        lw = eng._get_layer_weights(layer_idx)
        if lw.get("attn_norm.weight") is not None:
            h = rms_norm(h, lw["attn_norm.weight"].to(torch.bfloat16), eng.config.rms_norm_eps)

        h_mhc, _, _ = eng._process_mhc_layer(h, lw, "hc_attn")
        attn_out = eng._forward_attn(h if h_mhc is None else h_mhc, lw, layer_idx)
        h = h + attn_out

        if lw.get("ffn_norm.weight") is not None:
            h = rms_norm(h, lw["ffn_norm.weight"].to(torch.bfloat16), eng.config.rms_norm_eps)

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
            scores = torch.matmul(h.to(gate_w.dtype), gate_w.t())
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

        residual = h
        h_pre, _, _ = eng._process_mhc_layer(h, lw, "hc_ffn")
        h_ffn_in = h_pre if h_pre is not None else h

        ffn_out = eng._forward_ffn(h_ffn_in, lw, layer_idx, input_ids)
        h = h + ffn_out

        if (layer_idx + 1) % 10 == 0:
            print(f"[hot_expert]  Layer {layer_idx+1}/{eng.config.num_hidden_layers} done")

    total_calls = sum(expert_counts.values())
    top_16 = [eid for eid, _ in expert_counts.most_common(16)]
    top_16_coverage = sum(expert_counts[eid] for eid in top_16) / total_calls if total_calls > 0 else 0

    result = {
        "top_16_hot_experts": top_16,
        "top_16_coverage": round(top_16_coverage, 4),
        "total_expert_calls": total_calls,
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
    print(f"[hot_expert] Top-16 experts: {top_16}")
    print(f"[hot_expert] Top-16 coverage: {top_16_coverage:.2%}")
    print(f"[hot_expert] Saved to {output_path}")
    return result


if __name__ == "__main__":
    import sys
    weight_dir = sys.argv[1] if len(sys.argv) > 1 else "weights"
    output = sys.argv[2] if len(sys.argv) > 2 else "hot_experts.json"
    analyze_hot_experts(weight_dir, output_path=output)
