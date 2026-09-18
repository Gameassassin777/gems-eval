"""gems-eval command line.

  gems-eval holdout  --labels existing_faults.tif --out seghold/ [--frac 0.3 --seed 0 --buffer 3]
  gems-eval score    --pred pred.tif --eval-labels seghold/eval_labels.tif [--top 0.02] [--json]
  gems-eval trim     --pred pred.tif --template existing_faults.tif --out trimmed.tif --top 0.02 [--binarise]
  gems-eval validate --sub submission.tif --template existing_faults.tif
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from .holdout import split_segments, holdout_dti, summary
from .postprocess import keep_top_fraction
from .validate import validate_submission


def _read(path):
    import rasterio
    with rasterio.open(path) as s:
        return s.read(1), s.profile


def cmd_holdout(a):
    import rasterio
    y, prof = _read(a.labels)
    train, ev = split_segments(y, frac=a.frac, seed=a.seed, exclude_buffer_px=a.buffer)
    prof.update(dtype="int8", nodata=-1, count=1, compress="deflate")
    os.makedirs(a.out, exist_ok=True)
    for arr, name in ((train, "train_labels.tif"), (ev, "eval_labels.tif")):
        with rasterio.open(os.path.join(a.out, name), "w", **prof) as d:
            d.write(arr, 1)
    print(summary(ev)); print(f"training fault px {int((train == 1).sum())}; written to {a.out}")


def cmd_score(a):
    pred, _ = _read(a.pred); ev, _ = _read(a.eval_labels)
    valid = (ev == 0) | (ev == 1)
    p = np.nan_to_num(pred.astype(np.float32), nan=0.0)
    res = {"dti": holdout_dti(p, ev), "coverage_ge_0.5": float(np.mean(p[valid] >= 0.5)),
           "mean_pred": float(np.mean(p[valid]))}
    if a.top:
        res[f"dti_top{a.top:g}"] = holdout_dti(keep_top_fraction(p, a.top, valid=valid), ev)
    print(json.dumps(res, indent=1) if a.json else "\n".join(f"{k}: {v:.4f}" for k, v in res.items()))


def cmd_trim(a):
    import rasterio
    pred, prof = _read(a.pred); tmpl, _ = _read(a.template)
    inside = tmpl >= 0
    out = keep_top_fraction(np.nan_to_num(pred.astype(np.float32), nan=0.0), a.top, valid=inside, binarise=a.binarise)
    out[~inside] = np.nan
    prof.update(dtype="float32", nodata=np.nan, count=1, compress="deflate")
    with rasterio.open(a.out, "w", **prof) as d:
        d.write(out.astype(np.float32), 1)
    print(f"kept top {a.top:g} of valid pixels -> {a.out}; mean inside {np.nanmean(out[inside]):.4f}")


def cmd_validate(a):
    res = validate_submission(a.sub, a.template)
    print(json.dumps(res, indent=1)); sys.exit(0 if res["ok"] else 2)


def main(argv=None):
    p = argparse.ArgumentParser(prog="gems-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("holdout"); h.add_argument("--labels", required=True); h.add_argument("--out", required=True)
    h.add_argument("--frac", type=float, default=0.3); h.add_argument("--seed", type=int, default=0)
    h.add_argument("--buffer", type=int, default=3); h.set_defaults(f=cmd_holdout)
    s = sub.add_parser("score"); s.add_argument("--pred", required=True); s.add_argument("--eval-labels", required=True)
    s.add_argument("--top", type=float, default=0.02); s.add_argument("--json", action="store_true"); s.set_defaults(f=cmd_score)
    t = sub.add_parser("trim"); t.add_argument("--pred", required=True); t.add_argument("--template", required=True)
    t.add_argument("--out", required=True); t.add_argument("--top", type=float, default=0.02)
    t.add_argument("--binarise", action="store_true"); t.set_defaults(f=cmd_trim)
    v = sub.add_parser("validate"); v.add_argument("--sub", required=True); v.add_argument("--template", required=True)
    v.set_defaults(f=cmd_validate)
    a = p.parse_args(argv); a.f(a)


if __name__ == "__main__":
    main()
