import numpy as np

from baselines import persistence, seasonal_naive
from data_loader import Scaler, WindowDataset, chronological_split, generate_synthetic_weather


def test_splits_are_chronological_and_disjoint():
    tr, va, te = chronological_split(720)
    assert tr.stop == va.start and va.stop == te.start and te.stop == 720
    assert tr.start == 0


def test_scaler_uses_training_statistics_only():
    x = np.arange(100, dtype=float)[:, None]
    tr, _, _ = chronological_split(100)
    s = Scaler().fit(x[tr])
    assert np.isclose(s.mean_[0], x[tr].mean())
    assert not np.isclose(s.mean_[0], x.mean())


def test_window_targets_stay_inside_their_split_and_only_look_back():
    f, y = generate_synthetic_weather(200)
    _, va, _ = chronological_split(200)
    ds = WindowDataset(f, y, 24, range(va.start, va.stop))
    for k in range(len(ds)):
        j = ds.idx[k]
        x, t = ds[k]
        assert va.start <= j < va.stop
        assert np.allclose(x.numpy(), f[j - 24 : j])  # strictly before the target
        assert np.isclose(t.item(), y[j])


def test_baselines_are_one_step_and_causal():
    y = np.arange(100, dtype=float)
    idx = np.array([50, 60])
    assert (persistence(y, idx) == y[idx - 1]).all()
    assert (seasonal_naive(y, idx) == y[idx - 24]).all()


def test_synthetic_data_is_reproducible():
    a = generate_synthetic_weather(50, seed=1)[1]
    b = generate_synthetic_weather(50, seed=1)[1]
    assert np.array_equal(a, b)
