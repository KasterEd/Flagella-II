"""
inference.py — Full-volume sliding-window inference + motor peak detection.

Runs a trained 3-D U-Net over an entire tomogram using overlapping 128³ patches,
averages predictions in the overlap regions, then detects motor positions as
local maxima in the resulting heatmap.

Usage:
    python inference.py \\
        --tomo_id  TS_0001 \\
        --data_root  train/ \\
        --checkpoint checkpoints/best.pth \\
        --n_slices   300 \\
        --threshold  0.3

Output:
    <tomo_id>_heatmap.npy  — full-volume predicted heatmap
    Detected motor positions printed to stdout.
"""

import argparse
import os

import numpy as np
from PIL import Image
import torch
from scipy.ndimage import maximum_filter

from model import UNet3D


# ── Volume utilities ──────────────────────────────────────────────────────────

def load_and_normalise(tomo_dir: str, n_slices: int) -> np.ndarray:
    """Load all JPEG slices and apply the same percentile normalisation used in training."""
    slices = []
    for i in range(n_slices):
        path = os.path.join(tomo_dir, f"slice_{i:04d}.jpg")
        slices.append(np.array(Image.open(path), dtype=np.float32))
    volume = np.stack(slices, axis=0)   # (D, H, W)

    p_lo = np.percentile(volume, 0.5)
    p_hi = np.percentile(volume, 99.5)
    volume = np.clip(volume, p_lo, p_hi)
    volume = (volume - p_lo) / (p_hi - p_lo + 1e-8)
    return volume


# ── Sliding-window inference ──────────────────────────────────────────────────

def sliding_window_inference(
    model: torch.nn.Module,
    volume: np.ndarray,
    patch_size: int = 128,
    stride: int = 64,
    batch_size: int = 8,
    device: str = "cuda",
    use_amp: bool = True,
) -> np.ndarray:
    """
    Tile the full volume with overlapping patches, run the model on each,
    and average predictions in overlapping regions (Gaussian blending is
    replaced by simple average for simplicity; results are equivalent for
    detection).

    Args:
        model      : trained UNet3D (eval mode set internally).
        volume     : float32 numpy array (D, H, W), normalised.
        patch_size : edge length of cubic patch (must match training).
        stride     : step between patch origins; overlap = patch_size - stride.
                     stride=64 with patch_size=128 → 50 % overlap.
        batch_size : number of patches forwarded in one GPU call.
        device     : 'cuda' or 'cpu'.
        use_amp    : use bfloat16 autocast (A100).

    Returns:
        Averaged heatmap, same shape as `volume`, dtype float32.
    """
    model.eval()
    D, H, W = volume.shape
    P, S    = patch_size, stride

    pred_acc = np.zeros((D, H, W), dtype=np.float64)
    cnt_acc  = np.zeros((D, H, W), dtype=np.float64)

    def _starts(dim_size: int) -> list:
        """Patch origins that cover the full dimension including a final flush."""
        pts = list(range(0, dim_size - P, S))
        # Ensure the far edge is always covered
        if not pts or pts[-1] + P < dim_size:
            pts.append(dim_size - P)
        return pts

    origins = [
        (z0, y0, x0)
        for z0 in _starts(D)
        for y0 in _starts(H)
        for x0 in _starts(W)
    ]
    print(f"  Total patches: {len(origins):,}  (stride={S}, patch={P})")

    buf_patches: list = []
    buf_origins: list = []

    def _flush():
        if not buf_patches:
            return
        batch = torch.stack(buf_patches).to(device)   # (N, 1, P, P, P)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            preds = model(batch).squeeze(1).float().cpu().numpy()  # (N, P, P, P)
        for pred, (z0, y0, x0) in zip(preds, buf_origins):
            pred_acc[z0 : z0 + P, y0 : y0 + P, x0 : x0 + P] += pred
            cnt_acc [z0 : z0 + P, y0 : y0 + P, x0 : x0 + P] += 1
        buf_patches.clear()
        buf_origins.clear()

    for origin in origins:
        z0, y0, x0 = origin
        patch = volume[z0 : z0 + P, y0 : y0 + P, x0 : x0 + P].astype(np.float32)
        buf_patches.append(torch.from_numpy(patch).unsqueeze(0))
        buf_origins.append(origin)
        if len(buf_patches) == batch_size:
            _flush()
    _flush()   # remaining partial batch

    return (pred_acc / (cnt_acc + 1e-8)).astype(np.float32)


# ── Peak detection ────────────────────────────────────────────────────────────

def detect_motors(
    heatmap: np.ndarray,
    threshold: float = 0.3,
    nms_radius: int = 20,
) -> list:
    """
    Detect local maxima in the 3-D heatmap above `threshold`.

    Non-maximum suppression is implemented via scipy maximum_filter:
    a voxel is a peak iff it is the maximum within a cube of side
    (2*nms_radius + 1) centred on it.

    Args:
        heatmap    : float32 (D, H, W) array from sliding_window_inference.
        threshold  : minimum peak value (tune on validation set).
        nms_radius : half-size of the NMS window in voxels.

    Returns:
        List of (z, y, x) tuples sorted by descending score.
    """
    window  = 2 * nms_radius + 1
    loc_max = maximum_filter(heatmap, size=window)
    peaks   = (heatmap == loc_max) & (heatmap > threshold)
    coords  = np.argwhere(peaks)       # (N, 3)
    scores  = heatmap[peaks]
    order   = np.argsort(-scores)
    return [(int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2])) for i in order]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Sliding-window inference on a single tomogram.")
    ap.add_argument("--data_root",   default="train/",           help="Path to train/ directory")
    ap.add_argument("--tomo_id",     required=True,              help="Tomogram ID (sub-folder name)")
    ap.add_argument("--checkpoint",  default="checkpoints/best.pth")
    ap.add_argument("--patch_size",  type=int,   default=128)
    ap.add_argument("--stride",      type=int,   default=64,     help="Sliding-window stride (voxels)")
    ap.add_argument("--batch_size",  type=int,   default=8,      help="Patches per GPU forward pass")
    ap.add_argument("--threshold",   type=float, default=0.3,    help="Peak detection threshold")
    ap.add_argument("--nms_radius",  type=int,   default=20,     help="NMS half-window (voxels)")
    ap.add_argument("--n_slices",    type=int,   default=300,    help="Number of slices in this tomo")
    ap.add_argument("--no_amp",      action="store_true",        help="Disable bfloat16 autocast")
    args = ap.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"Device  : {device}  (amp={'bfloat16' if use_amp else 'off'})")

    # ── Load checkpoint & model ───────────────────────────────────────────────
    ckpt  = torch.load(args.checkpoint, map_location=device)
    cfg   = ckpt.get("cfg", {})
    model = UNet3D(in_ch=1, out_ch=1, f=cfg.get("base_features", 32)).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}  "
          f"(val_loss={ckpt.get('val_loss', float('nan')):.6f})")

    # ── Load volume ──────────────────────────────────────────────────────────
    tomo_dir = os.path.join(args.data_root, args.tomo_id)
    print(f"Loading volume from {tomo_dir} …")
    volume = load_and_normalise(tomo_dir, args.n_slices)
    print(f"Volume shape : {volume.shape}   "
          f"(~{volume.nbytes / 1e9:.2f} GB as float32)")

    # ── Inference ────────────────────────────────────────────────────────────
    print("Running sliding-window inference …")
    heatmap = sliding_window_inference(
        model,
        volume,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
        device=str(device),
        use_amp=use_amp,
    )
    out_path = f"{args.tomo_id}_heatmap.npy"
    np.save(out_path, heatmap)
    print(f"Heatmap saved to {out_path}")
    print(f"Heatmap range  : [{heatmap.min():.4f}, {heatmap.max():.4f}]")

    # ── Detection ────────────────────────────────────────────────────────────
    motors = detect_motors(heatmap, threshold=args.threshold, nms_radius=args.nms_radius)
    print(f"\nDetected {len(motors)} motor(s) at threshold={args.threshold}:")
    for i, (z, y, x) in enumerate(motors):
        score = heatmap[z, y, x]
        print(f"  [{i}]  z={z:4d}  y={y:4d}  x={x:4d}  score={score:.4f}")


if __name__ == "__main__":
    main()