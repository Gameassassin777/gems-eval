import numpy as np
from gems_eval import dti, keep_top_fraction


def _scene():
    y = np.zeros((256, 256), bool)
    y[100, 20:230] = True
    y[30:220, 180] = True
    return y


def test_perfect_prediction_scores_one():
    y = _scene()
    assert dti(y.astype(np.float32), y) > 0.999


def test_blank_scores_zero_and_everywhere_is_low():
    y = _scene()
    assert dti(np.zeros(y.shape, np.float32), y) < 1e-6
    assert dti(np.ones(y.shape, np.float32), y) < 0.1


def test_partial_credit_decays_with_distance():
    y = np.zeros((256, 256), bool); y[100, 20:230] = True          # one horizontal fault
    p = y.astype(np.float32)
    s0 = dti(p, y); s2 = dti(np.roll(p, 2, axis=0), y); s10 = dti(np.roll(p, 10, axis=0), y)
    assert s0 > s2 > s10
    # 2 px off: k = 1/3 -> TP = N/3, FP = 2N/3, FN = 2N/3 -> (1/3)/(1/3 + 0.2*2/3 + 0.8*2/3) = 1/3
    assert abs(s2 - 1 / 3) < 0.01 and s10 < 0.01


def test_thick_band_earns_no_extra_credit_but_pays_false_positives():
    y = np.zeros((128, 128), bool); y[64, 10:120] = True
    from scipy import ndimage
    line = y.astype(np.float32)
    band3 = ndimage.binary_dilation(y, iterations=1).astype(np.float32)
    s_line, t_line = dti(line, y, return_terms=True)
    s_band, t_band = dti(band3, y, return_terms=True)
    assert abs(t_line["tp"] - t_band["tp"]) < 1e-3 and t_band["fp"] > t_line["fp"]
    assert s_line > 0.999 and 0.85 < s_band < 0.9        # 3-px band: 1/(1 + 0.2*2/3)


def test_halving_confidence_halves_tp_without_threshold():
    y = np.zeros((64, 64), bool); y[32, 5:60] = True
    s, t = dti(0.5 * y.astype(np.float32), y, return_terms=True)
    assert abs(t["tp"] - 0.5 * t["n_truth"]) < 1e-3 and abs(s - 0.5 / 0.9) < 1e-3


def test_valid_mask_excludes_pixels():
    y = _scene(); p = y.astype(np.float32)
    valid = np.ones(y.shape, bool); valid[:, :128] = False
    # left half excluded: still perfect on the right half
    assert dti(p, y, valid=valid) > 0.999
    # false positives in the excluded half cost nothing
    p2 = p.copy(); p2[:, :128] = 1.0
    assert abs(dti(p2, y, valid=valid) - dti(p, y, valid=valid)) < 1e-6


def test_keep_top_fraction_keeps_expected_share():
    rng = np.random.default_rng(0); p = rng.random((100, 100)).astype(np.float32)
    p[0, 0] = np.nan
    out = keep_top_fraction(p, 0.1)
    assert np.isnan(out[0, 0])
    assert abs(np.mean(out[np.isfinite(out)] > 0) - 0.1) < 0.005
    outb = keep_top_fraction(p, 0.1, binarise=True)
    assert set(np.unique(outb[np.isfinite(outb)])) <= {0.0, 1.0}


def test_skeleton_top_fraction_thins_to_lines():
    from gems_eval import skeleton_top_fraction
    y = np.zeros((128, 128), bool); y[64, 10:120] = True
    from scipy import ndimage
    band = ndimage.binary_dilation(y, iterations=2).astype(np.float32) * 0.9      # 5-px wide band
    out = skeleton_top_fraction(band, 0.05)
    assert out[np.isfinite(out)].sum() < 1.5 * y.sum()                           # thinned to ~one line
    assert dti(out, y) > dti(band, y)                                            # and it scores better
