"""
train.py — Training entry point for 3-D cryo-EM heatmap regression.

Run:
    python train.py

Then open TensorBoard in a second terminal:
    tensorboard --logdir runs/

All hyperparameters live in CFG at the top of this file.

A100-specific optimisations used:
  - bfloat16 autocast (same exponent range as float32 → no GradScaler needed)
  - persistent_workers=True (workers stay alive between epochs;
    volume cache built in epoch 1 is reused for free in all subsequent epochs)
  - pin_memory=True + non_blocking transfers

TensorBoard panels written:
  Scalars   │ Loss/train, Loss/val, LearningRate           — every epoch
  Images    │ train/input_patch, train/target_heatmap,     — every viz_every epochs
             │ train/pred_heatmap, val/input_patch,
             │ val/target_heatmap, val/pred_heatmap
             │ (central axial slice of the first patch in each batch)
  Histograms│ model weight and gradient distributions       — every viz_every epochs
  Text      │ run config (CFG) + train/val tomo split       — once at start
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataLoader import CryoEMPatchDataset
from model import UNet3D
from torch.utils.tensorboard import SummaryWriter
# ── Configuration ─────────────────────────────────────────────────────────────
CFG = dict(
    # Paths
    data_root      = "/data/quokka/ws/0/kaku169g-flagella-data/data/train/",
    labels_csv     = "/data/quokka/ws/0/kaku169g-flagella-data/data/train_labels.csv",
    checkpoint_dir = "/data/horse/ws/kaku169g-flagella2/Flagella-II/checkpoints/",

    # Data
    patch_size        = 128,
    patches_per_tomo  = 100,   # number of patches drawn per tomogram per epoch
    positive_ratio    = 0.6,   # fraction of patches centred near a motor
    base_sigma_px     = 5.0,   # Gaussian sigma at ref_spacing=10 Å/px
    ref_spacing       = 10.0,  # Å/px — reference for sigma normalisation
    scale_sigma       = True,  # scale sigma by voxel spacing
    val_fraction      = 0.2,
    seed              = 42,

    # Training
    batch_size    = 4,
    num_workers   = 4,
    lr            = 1e-4,
    weight_decay  = 1e-5,
    epochs        = 100,

    # Model
    base_features = 32,   # U-Net feature multiplier

    # Hardware
    use_amp       = True,  # bfloat16 on A100
)


# ── Epoch runners ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, use_amp):
    model.train()
    total_loss = 0.0

    for patches, heatmaps in tqdm(loader, desc="  train", leave=False):
        patches  = patches.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            pred = model(patches)
            loss = F.mse_loss(pred, heatmaps)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device, use_amp):
    model.eval()
    total_loss = 0.0

    for patches, heatmaps in tqdm(loader, desc="  val  ", leave=False):
        patches  = patches.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            pred = model(patches)
            loss = F.mse_loss(pred, heatmaps)

        total_loss += loss.item()

    return total_loss / len(loader)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)
    run_name = f"unet3d_seed{CFG['seed']}"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    # ── Data split (by tomo_id — NEVER split individual slices) ─────────────
    df      = pd.read_csv(CFG["labels_csv"])
    all_ids = df["tomo_id"].unique().tolist()

    rng = np.random.default_rng(CFG["seed"])
    rng.shuffle(all_ids)
    n_val     = max(1, int(len(all_ids) * CFG["val_fraction"]))
    val_ids   = all_ids[:n_val]
    train_ids = all_ids[n_val:]
    print(f"Split  : {len(train_ids)} train tomos / {len(val_ids)} val tomos")

    # Persist the split so results are reproducible across runs
    split_path = os.path.join(CFG["checkpoint_dir"], "split.json")
    with open(split_path, "w") as fh:
        json.dump({"train": train_ids, "val": val_ids}, fh, indent=2)

    # ── Datasets & loaders ───────────────────────────────────────────────────
    shared_ds_kwargs = dict(
        df                    = df,
        data_root             = CFG["data_root"],
        patch_size            = CFG["patch_size"],
        positive_ratio        = CFG["positive_ratio"],
        sigma_px              = CFG["base_sigma_px"],
        ref_spacing           = CFG["ref_spacing"],
        scale_sigma           = CFG["scale_sigma"],
        patches_per_tomo      = CFG["patches_per_tomo"],
    )
    train_ds = CryoEMPatchDataset(train_ids, augment=True,  **shared_ds_kwargs)
    val_ds   = CryoEMPatchDataset(val_ids,   augment=False, **shared_ds_kwargs)

    shared_loader_kwargs = dict(
        batch_size        = CFG["batch_size"],
        num_workers       = CFG["num_workers"],
        pin_memory        = True,
        # persistent_workers keeps worker processes alive between epochs so the
        # volume cache (built during epoch 1) is reused rather than reloaded.
        persistent_workers = CFG["num_workers"] > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **shared_loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **shared_loader_kwargs)

    # ── Model, optimiser, scheduler ──────────────────────────────────────────
    model = UNet3D(in_ch=1, out_ch=1, f=CFG["base_features"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params : {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"]
    )

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, CFG["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, CFG["use_amp"])
        val_loss   = validate(model, val_loader, device, CFG["use_amp"])
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        print(
            f"Epoch {epoch:4d}/{CFG['epochs']}  │  "
            f"train {train_loss:.6f}  │  val {val_loss:.6f}  │  lr {current_lr:.2e}"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(CFG["checkpoint_dir"], "best.pth")
            torch.save(
                {
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss":        val_loss,
                    "cfg":             CFG,
                },
                ckpt_path,
            )
            print(f"           ↳ saved best checkpoint  (val_loss={val_loss:.6f})")

    # Save training history for plotting
    history_path = os.path.join(CFG["checkpoint_dir"], "history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    print(f"History saved to {history_path}")
    writer.close()


if __name__ == "__main__":
    main()