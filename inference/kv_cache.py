import torch


class KVCache:
    """手写 KV Cache：预分配连续 buffer + 位置游标。

    prefill 一次写入 prompt 全部位置的 K/V；decode 每步只 append 最后一个
    token 的 K/V，其余位置复用，避免重算历史 KV。
    """

    def __init__(self, num_layers, batch, num_kv_heads, head_dim, max_seq_len,
                 dtype=torch.float32, device="cpu"):
        self.num_layers = num_layers
        self.batch = batch
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device
        shape = (batch, num_kv_heads, max_seq_len, head_dim)
        self._keys = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self._values = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self._seq_len = 0

    @property
    def seq_len(self):
        return self._seq_len

    def update(self, layer_idx, key, value, positions):
        """写入 key/value（shape (batch, num_kv_heads, len(positions), head_dim)）到 positions 指定位置。"""
        self._keys[layer_idx][:, :, positions] = key
        self._values[layer_idx][:, :, positions] = value

    def get(self, layer_idx):
        """返回当前有效 KV 切片，各 (batch, num_kv_heads, seq_len, head_dim)。"""
        return (self._keys[layer_idx][:, :, : self._seq_len],
                self._values[layer_idx][:, :, : self._seq_len])

    def advance(self, n):
        self._seq_len += n
