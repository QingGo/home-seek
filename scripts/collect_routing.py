"""Use engine.generate() + monkey-patch _forward_ffn to collect per-layer routing."""
import json
from collections import Counter, defaultdict
import torch

from home_seek.inference_engine import HomeSeekInferenceEngine


@torch.no_grad()
def main():
    prompts = [
        "Hello",
        "What is AI?",
        "Write a poem",
        "How are you?",
        "Hi",
    ]

    eng = HomeSeekInferenceEngine("weights", verbose=False)

    global_counts = Counter()
    layer_trace = defaultdict(list)

    original_ffn = eng._forward_ffn
    def _collecting_ffn(hidden_states, lw, layer_idx, input_ids=None):
        result, used = original_ffn(hidden_states, lw, layer_idx, input_ids)
        if hasattr(eng, '_last_routed_eids'):
            for eid in eng._last_routed_eids:
                if eid >= 0:
                    global_counts[eid] += 1
                    layer_trace[layer_idx].append(eid)
        return result, used

    eng._forward_ffn = _collecting_ffn

    for prompt_text in prompts:
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file="weights/tokenizer.json")
        input_ids = tok.encode(prompt_text)
        input_ids = torch.tensor([input_ids], dtype=torch.long)

        generated = []
        list(eng.generate(input_ids, max_new_tokens=12, temperature=0,
                          stream_callback=lambda tid: generated.append(tid)))
        torch.cuda.empty_cache()

    eng._forward_ffn = original_ffn

    # Compute per-layer hot sets
    per_layer_hot_count = 32
    global_hot_count = 64
    top_hot_by_layer = {}
    for layer_idx in sorted(layer_trace.keys()):
        layer_counter = Counter(layer_trace[layer_idx])
        layer_top = [eid for eid, _ in layer_counter.most_common(per_layer_hot_count)]
        top_hot_by_layer[str(layer_idx)] = layer_top

    global_top = [eid for eid, _ in global_counts.most_common(global_hot_count)]

    result = {
        "top_hot_experts": global_top,
        "num_hot_experts": global_hot_count,
        "top_hot_experts_by_layer": top_hot_by_layer,
        "per_layer_hot_count": per_layer_hot_count,
        "config": {
            "num_layers": eng.config.num_hidden_layers,
            "num_experts": eng.config.n_routed_experts,
            "num_hash_layers": eng.config.num_hash_layers,
        },
    }

    with open("hot_experts.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Total routing observations: {sum(global_counts.values())}")
    print(f"Unique experts seen: {len(global_counts)}")
    print(f"Global top-{global_hot_count}: {global_top}")
    print(f"Per-layer hot: {len(top_hot_by_layer)} layers × up to {per_layer_hot_count} experts")
    print("Saved to hot_experts.json")


if __name__ == "__main__":
    main()
