"""Quick diagnostic for pre-extracted 3D-JEPA features.

Run when the visual ablation in eval shows that zeroing JEPA features barely
changes accuracy — that means the model is not using visual signal, but it
does NOT yet tell you whether the cause is (A) the features themselves are
uninformative / corrupt, or (B) the model fails to learn from informative
features. This script answers (A).

Usage:
    python scripts/3d/inspect_jepa_features.py
    python scripts/3d/inspect_jepa_features.py --folder data/3d-jepa-features --num 5

Outputs per-scene statistics + a cross-scene comparison. Healthy features
should have:
  * per-point feature std > 0 (points within a scene differ from each other)
  * scene-level mean/std varying across scenes (different scenes look different)
  * coords spanning a sensible 3D range (e.g., a few meters)

If per-point std ≈ 0, all points carry the same vector → features are
collapsed and useless. If scene-level stats are nearly identical across
scenes, features encode no scene-specific information.
"""

import argparse
import glob
import os

import torch


def load_one(path: str):
    """Return (features, coords) regardless of the .pt's container shape."""
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        feats = data.get("features", data.get("jepa_features"))
        coords = data.get("coords", data.get("points"))
        if feats is None or coords is None:
            raise ValueError(f"{path}: dict missing features/coords keys ({list(data.keys())})")
        return feats, coords
    if isinstance(data, (tuple, list)) and len(data) >= 2:
        return data[0], data[1]
    raise ValueError(f"{path}: unsupported container type {type(data)}")


def stats_for_scene(feats: torch.Tensor, coords: torch.Tensor) -> dict:
    f = feats.float()
    c = coords.float()
    return {
        "n": int(f.shape[0]),
        "feat_dim": int(f.shape[1]),
        "feat_mean": f.mean().item(),
        "feat_std": f.std().item(),
        "feat_abs_mean": f.abs().mean().item(),
        # Average across points of the per-feature-dim std within a single
        # point — if this is ~0 every point has the same vector, collapse.
        "per_point_std": f.std(dim=1).mean().item(),
        # std across points of the per-dim mean — if this is ~0, all points
        # share the same value in each dim (another collapse signature).
        "across_point_std": f.std(dim=0).mean().item(),
        "any_nan": bool(torch.isnan(f).any().item()),
        "coord_min": c.min(dim=0).values.tolist(),
        "coord_max": c.max(dim=0).values.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="data/3d-jepa-features",
                    help="Folder of <scene_id>.pt JEPA feature files.")
    ap.add_argument("--num", type=int, default=5,
                    help="Number of scenes to inspect.")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.folder, "*.pt")))
    if not paths:
        raise SystemExit(f"No .pt files in {args.folder}")
    paths = paths[: args.num]

    rows = []
    for p in paths:
        feats, coords = load_one(p)
        s = stats_for_scene(feats, coords)
        s["scene"] = os.path.basename(p).replace(".pt", "")
        rows.append(s)

    print(f"\n=== Per-scene stats ({len(rows)} scenes from {args.folder}) ===")
    for s in rows:
        print(
            f"{s['scene']:>20s} | "
            f"N={s['n']:>6d} dim={s['feat_dim']:>4d} | "
            f"mean={s['feat_mean']:+.4f} std={s['feat_std']:.4f} "
            f"abs_mean={s['feat_abs_mean']:.4f} | "
            f"per_point_std={s['per_point_std']:.4f} "
            f"across_point_std={s['across_point_std']:.4f} | "
            f"nan={s['any_nan']} | "
            f"coord x:[{s['coord_min'][0]:.1f},{s['coord_max'][0]:.1f}] "
            f"y:[{s['coord_min'][1]:.1f},{s['coord_max'][1]:.1f}] "
            f"z:[{s['coord_min'][2]:.1f},{s['coord_max'][2]:.1f}]"
        )

    print("\n=== Cross-scene variation (does each scene look different?) ===")
    means = torch.tensor([r["feat_mean"] for r in rows])
    stds = torch.tensor([r["feat_std"] for r in rows])
    print(f"scene-mean    range=[{means.min():+.4f},{means.max():+.4f}]  std_across_scenes={means.std():.4f}")
    print(f"scene-std     range=[{stds.min():.4f},{stds.max():.4f}]  std_across_scenes={stds.std():.4f}")

    print("\n=== Verdict ===")
    issues = []
    pps = [r["per_point_std"] for r in rows]
    if min(pps) < 1e-3:
        issues.append(
            "per_point_std near 0 — points within a scene carry (almost) the same vector. "
            "Features are collapsed; visual signal is meaningless."
        )
    if means.std().item() < 1e-3 and stds.std().item() < 1e-3:
        issues.append(
            "scene-level mean/std barely vary across scenes — features encode no scene-specific info."
        )
    if any(r["any_nan"] for r in rows):
        issues.append("NaN detected in features — extraction is broken.")
    if not issues:
        print("✅ Features look structured and scene-specific. The bottleneck is on the model side: "
              "raise jepa_projector LR, increase LoRA r/alpha, or increase jepa_max_tokens.")
    else:
        print("❌ Problems found:")
        for i, msg in enumerate(issues, 1):
            print(f"  {i}. {msg}")
        print("Fix: re-extract JEPA features. Model-side hyperparam tuning will not help.")


if __name__ == "__main__":
    main()
