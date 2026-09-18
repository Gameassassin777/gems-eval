import numpy as np
from gems_eval import split_segments, holdout_dti


def test_split_codes_and_disjointness():
    rng = np.random.default_rng(0)
    y = np.zeros((300, 300), np.int8)
    for k in range(20):                      # 20 short horizontal segments
        r = 10 + 14 * k; c = rng.integers(0, 200); y[r, c:c + 60] = 1
    y[:5, :] = -1                            # a nodata strip
    train, ev = split_segments(y, frac=0.3, seed=1, exclude_buffer_px=3)
    assert set(np.unique(ev)) <= {-1, 0, 1, 2} and set(np.unique(train)) <= {-1, 0, 1}
    assert (ev[:5] == -1).all() and (train[:5] == -1).all()
    assert not ((train == 1) & (ev == 1)).any()          # held-out never in training
    assert ((ev == 2) >= (train == 1)).all()             # every training fault sits in the buffer
    assert 4 <= (ev == 1).sum() / 60 <= 8                # ~30% of 20 segments held out


def test_holdout_dti_ignores_training_faults():
    y = np.zeros((200, 200), np.int8); y[50, 20:180] = 1; y[150, 20:180] = 1
    train, ev = split_segments(y, frac=0.5, seed=0)
    held = (ev == 1).astype(np.float32)
    assert holdout_dti(held, ev) > 0.999
    both = (y == 1).astype(np.float32)                   # predicting the training fault too costs nothing
    assert abs(holdout_dti(both, ev) - holdout_dti(held, ev)) < 1e-6
