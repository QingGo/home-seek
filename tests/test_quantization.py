import torch
import pytest


class TestQuantization:
    def setup_method(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_fp4_roundtrip_cosine_similarity(self):
        from tile_reference import cast, unpack_from_e2m1fn_x2

        torch.manual_seed(42)
        for h in [64, 128, 256, 512]:
            for w in [128, 256, 512]:
                x = torch.randn(h, w, device="cuda", dtype=torch.bfloat16)
                quantized, sf = cast(x, fmt="e2m1", block_size=(1, 32))

                deq = unpack_from_e2m1fn_x2(quantized)
                sf_expanded = sf.repeat_interleave(32, dim=1)
                deq = deq.to(torch.float32) * sf_expanded.to(torch.float32)
                deq = deq[:x.shape[0], :x.shape[1]]

                cos_sim = torch.nn.functional.cosine_similarity(
                    x.flatten().unsqueeze(0).float(),
                    deq.flatten().unsqueeze(0),
                ).item()
                mse = ((x.float() - deq) ** 2).mean().item()
                assert cos_sim >= 0.99, f"h={h},w={w}: cos_sim={cos_sim:.6f}, mse={mse:.6e}"

    def test_fp8_roundtrip(self):
        from tile_reference import cast, cast_back

        torch.manual_seed(42)
        x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        q, sf = cast(x, fmt="e4m3", block_size=(32, 32))
        deq = cast_back((q, sf), fmt="fp32", block_size=(32, 32))
        deq = deq[:x.shape[0], :x.shape[1]]

        cos_sim = torch.nn.functional.cosine_similarity(
            x.flatten().unsqueeze(0).float(),
            deq.flatten().unsqueeze(0),
        ).item()
        assert cos_sim >= 0.995, f"FP8 roundtrip cos_sim too low: {cos_sim}"

    def test_expert_weight_approximation(self):
        from tile_reference import cast, unpack_from_e2m1fn_x2

        hidden = 4096
        intermediate = 2048
        torch.manual_seed(42)

        gate = torch.randn(intermediate, hidden, device="cuda", dtype=torch.bfloat16)
        up = torch.randn(intermediate, hidden, device="cuda", dtype=torch.bfloat16)
        down = torch.randn(hidden, intermediate, device="cuda", dtype=torch.bfloat16)

        q_gate, sf_gate = cast(gate, fmt="e2m1", block_size=(1, 32))
        q_up, sf_up = cast(up, fmt="e2m1", block_size=(1, 32))
        q_down, sf_down = cast(down, fmt="e2m1", block_size=(1, 32))

        def deq_fp4(data, sf):
            d = unpack_from_e2m1fn_x2(data)
            sf_e = sf.repeat_interleave(32, dim=1)
            d = d.to(torch.float32) * sf_e.to(torch.float32)
            return d[:data.shape[0] * 2, :sf.shape[1] * 32][:data.shape[0]]

        for name, orig, qd, sfd in [
            ("gate", gate, q_gate, sf_gate),
            ("up", up, q_up, sf_up),
            ("down", down, q_down, sf_down),
        ]:
            cos_sim = torch.nn.functional.cosine_similarity(
                orig.flatten().unsqueeze(0).float(),
                deq_fp4(qd, sfd)[:orig.shape[0], :orig.shape[1]].flatten().unsqueeze(0),
            ).item()
            assert cos_sim >= 0.99, f"{name}: cos_sim={cos_sim:.6f}"
