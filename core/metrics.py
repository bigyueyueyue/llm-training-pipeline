def ema_smooth(values: list[float], window: int = 10) -> list[float]:
    """指数滑动平均，平滑 loss 曲线。"""
    if not values:
        return []
    alpha = 2 / (window + 1)
    out, ema = [], None
    for v in values:
        ema = v if ema is None else alpha * v + (1 - alpha) * ema
        out.append(ema)
    return out


def steps_to_threshold(losses: list[float], threshold: float, window: int = 10) -> int | None:
    """平滑后首个 <= threshold 的步下标；未达阈值返回 None。"""
    for i, v in enumerate(ema_smooth(losses, window=window)):
        if v <= threshold:
            return i
    return None
