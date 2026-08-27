import torch
import pytest

from inference.kv_cache import KVCache


@pytest.fixture
def cache():
    # num_layers=2, batch=1, num_kv_heads=4, head_dim=8, max_seq_len=16
    return KVCache(num_layers=2, batch=1, num_kv_heads=4, head_dim=8,
                   max_seq_len=16, dtype=torch.float32, device="cpu")


def test_init_shapes(cache):
    for layer in range(cache.num_layers):
        k, v = cache.get(layer)
        # get() 返回按 seq_len 裁剪的有效切片；初始 seq_len=0 时为空切片
        assert k.shape == (1, 4, 0, 8)
        assert v.shape == (1, 4, 0, 8)
        assert k.dtype == torch.float32
    assert cache.seq_len == 0


def test_prefill_write_read_roundtrip(cache):
    L = 5
    k = torch.randn(1, 4, L, 8)
    v = torch.randn(1, 4, L, 8)
    cache.update(0, k, v, torch.arange(L))
    cache.advance(L)
    got_k, got_v = cache.get(0)
    assert got_k.shape == (1, 4, L, 8)
    assert torch.allclose(got_k, k)
    assert torch.allclose(got_v, v)


def test_decode_incremental_append(cache):
    L = 5
    k_pre = torch.randn(1, 4, L, 8)
    v_pre = torch.randn(1, 4, L, 8)
    cache.update(0, k_pre, v_pre, torch.arange(L))
    cache.advance(L)
    k_dec = torch.randn(1, 4, 1, 8)
    v_dec = torch.randn(1, 4, 1, 8)
    cache.update(0, k_dec, v_dec, torch.tensor([L]))
    cache.advance(1)
    got_k, got_v = cache.get(0)
    assert got_k.shape == (1, 4, L + 1, 8)
    assert torch.allclose(got_k[:, :, :L], k_pre)
    assert torch.allclose(got_k[:, :, L:], k_dec)
    assert torch.allclose(got_v[:, :, L:], v_dec)


def test_get_respects_seq_len(cache):
    L = 3
    k = torch.randn(1, 4, L, 8)
    cache.update(0, k, torch.zeros_like(k), torch.arange(L))
    cache.advance(L)
    # 越过当前 seq_len=3 写第 6 位，get 不应返回它
    extra = torch.randn(1, 4, 1, 8)
    cache.update(0, extra, extra, torch.tensor([6]))
    got_k, _ = cache.get(0)
    assert got_k.shape == (1, 4, 3, 8)
    assert torch.allclose(got_k, k)


def test_update_at_noncontiguous_positions(cache):
    positions = torch.tensor([0, 2, 4])
    k = torch.randn(1, 4, 3, 8)
    cache.update(1, k, k, positions)
    cache.advance(5)
    got_k, _ = cache.get(1)
    assert torch.allclose(got_k[:, :, 0], k[:, :, 0])
    assert torch.allclose(got_k[:, :, 2], k[:, :, 1])
    assert torch.allclose(got_k[:, :, 4], k[:, :, 2])
    assert torch.all(got_k[:, :, 1] == 0)
    assert torch.all(got_k[:, :, 3] == 0)


def test_layer_isolation(cache):
    k0 = torch.randn(1, 4, 1, 8)
    cache.update(0, k0, k0, torch.tensor([0]))
    cache.advance(1)
    got_k1, _ = cache.get(1)
    assert torch.all(got_k1 == 0)
