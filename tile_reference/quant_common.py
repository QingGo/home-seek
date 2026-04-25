import torch
from tile_reference.quant_types import QuantTensor
from tile_reference.utils import ceil_div, align


def unpack_from_e2m1fn_x2(x: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    assert x.dtype == torch.int8 or x.dtype == torch.uint8
    if x.ndim == 0:
        raise ValueError('x must have at least 1 dimension so the last dim can be doubled')
    lo = (x & 0x0F).to(torch.int16)
    hi = ((x >> 4) & 0x0F).to(torch.int16)

    def decode_fp4_e2m1(n: torch.Tensor) -> torch.Tensor:
        s = (n >> 3) & 0x1
        e = (n >> 1) & 0x3
        m = n & 0x1
        sign = torch.where(s == 1, torch.tensor(-1.0, device=n.device), torch.tensor(1.0, device=n.device))
        bias = 1
        sub = (m.to(torch.float32) * 0.5) * (2.0 ** (1 - bias))
        norm = (1.0 + m.to(torch.float32) * 0.5) * torch.pow(torch.tensor(2.0, device=n.device), (e - bias).to(torch.float32))
        val = torch.where(e == 0, sub, norm)
        return (val * sign).to(out_dtype)

    flo = decode_fp4_e2m1(lo)
    fhi = decode_fp4_e2m1(hi)
    y = torch.stack([flo, fhi], dim=-1).reshape(*x.shape[:-1], x.shape[-1] * 2)
    return y


def transform_sf(sf: torch.Tensor) -> torch.Tensor:
    if sf.dtype == torch.float32:
        return sf
    assert sf.dtype == torch.int32
    sf = sf.contiguous()
    if sf.stride(-1) != 1:
        sf = sf.as_strided(size=sf.shape, stride=(sf.shape[-1], 1))
    sf = sf.view(torch.uint8)
    sf = sf.to(torch.int32)
    sf = (sf << 23).view(torch.float32)
    return sf
