"""吞吐压测（GPU-deferred）。

上卡后运行，测量 prefill（批量前向）与 decode（单 token 增量）两阶段吞吐，
并打印实际命中的注意力后端，确保 FA2 真跑了而非静默降级。

用法：python -m inference.benchmark --batch 1 2 4 8 --seq-len 128 256 512 1024 2048
"""
import argparse
import time

import torch

from inference.attention import attention_forward
from inference.kv_cache import KVCache


def _gqa_inputs(batch, seq_len, num_heads, num_kv_heads, head_dim, dtype, device):
    q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    return q, k, v


def bench_prefill(batch, seq_len, num_heads, num_kv_heads, head_dim, dtype, device, repeats=10):
    q, k, v = _gqa_inputs(batch, seq_len, num_heads, num_kv_heads, head_dim, dtype, device)
    for _ in range(3):
        attention_forward(q, k, v, num_kv_heads, is_causal=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        attention_forward(q, k, v, num_kv_heads, is_causal=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return batch * seq_len / (dt / repeats)  # tokens/s


def bench_decode(batch, seq_len, num_heads, num_kv_heads, head_dim, dtype, device, repeats=10):
    cache = KVCache(num_layers=1, batch=batch, num_kv_heads=num_kv_heads,
                    head_dim=head_dim, max_seq_len=seq_len + 1, dtype=dtype, device=device)
    k_pre = torch.randn(batch, num_kv_heads, seq_len - 1, head_dim, dtype=dtype, device=device)
    v_pre = torch.randn(batch, num_kv_heads, seq_len - 1, head_dim, dtype=dtype, device=device)
    cache.update(0, k_pre, v_pre, torch.arange(seq_len - 1, device=device))
    cache.advance(seq_len - 1)
    q = torch.randn(batch, num_heads, 1, head_dim, dtype=dtype, device=device)
    kk = torch.randn(batch, num_kv_heads, 1, head_dim, dtype=dtype, device=device)
    vv = torch.randn(batch, num_kv_heads, 1, head_dim, dtype=dtype, device=device)
    cache.update(0, kk, vv, torch.tensor([seq_len - 1], device=device))
    cache.advance(1)
    k_full, v_full = cache.get(0)  # (batch, num_kv_heads, seq_len, head_dim)
    # 新 token 对全量缓存历史做 full attention（query 即最后位置，无因果掩码）
    for _ in range(3):
        attention_forward(q, k_full, v_full, num_kv_heads, is_causal=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        attention_forward(q, k_full, v_full, num_kv_heads, is_causal=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return 1.0 / (dt / repeats)  # tokens/s（单 token 步）


def main(argv=None):
    parser = argparse.ArgumentParser(description="FA2/KV Cache 吞吐压测（GPU）")
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seq-len", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--num-heads", type=int, default=28)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    q, k, v = _gqa_inputs(1, 16, args.num_heads, args.num_kv_heads, args.head_dim, dtype, device)
    _, backend = attention_forward(q, k, v, args.num_kv_heads, is_causal=True)
    print(f"backend: {backend}")

    print("\nprefill tokens/s (batch x seq_len):")
    for b in args.batch:
        cells = [f"{s}:{bench_prefill(b, s, args.num_heads, args.num_kv_heads, args.head_dim, dtype, device):.0f}"
                 for s in args.seq_len]
        print(f"  batch={b}: " + "  ".join(cells))

    print("\ndecode tokens/s (batch, cache grows to seq_len):")
    for b in args.batch:
        for s in args.seq_len:
            t = bench_decode(b, s, args.num_heads, args.num_kv_heads, args.head_dim, dtype, device)
            print(f"  batch={b} seq_len={s}: {t:.1f} tokens/s")


if __name__ == "__main__":
    main()
