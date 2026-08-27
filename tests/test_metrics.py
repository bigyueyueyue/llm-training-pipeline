from core.metrics import ema_smooth, steps_to_threshold


def test_steps_to_threshold():
    losses = [3.0, 2.5, 2.0, 1.8, 1.5]
    assert steps_to_threshold(losses, threshold=1.7, window=1) == 4


def test_steps_to_threshold_none():
    losses = [3.0, 2.5, 2.0]
    assert steps_to_threshold(losses, threshold=1.0, window=1) is None


def test_ema_smooth_empty():
    assert ema_smooth([]) == []
