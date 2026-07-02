import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class CryoEMPatchDataset(Dataset):
    """
    Args:
        tomo_ids          : list of tomo_id strings assigned to this split.
        df                : full train_labels.csv DataFrame.
        data_root         : path to the train/ directory (contains <tomo_id>/ sub-dirs).
        patch_size        : edge length of cubic 3-D patch (default 128).
        positive_ratio    : fraction of __getitem__ calls that return a motor-centred patch.
        sigma_px          : Gaussian sigma in pixels at the reference voxel spacing.
        ref_spacing       : reference spacing (Å/px) used to normalise sigma (default 10.0).
        scale_sigma       : if True, sigma = sigma_px * (ref_spacing / voxel_spacing).
        patches_per_tomo  : controls dataset length = len(tomo_ids) × patches_per_tomo.
        augment           : apply random flips and 90° xy-rotations.
    """

    def __init__(
            self,
            tomo_ids: list,
            df: pd.DataFrame,
            data_root: str,
            patch_size: int = 128,
            positive_ratio: float = 0.6,
            sigma_px: float = 5.0,
            ref_spacing: float = 10.0,
            scale_sigma: bool = True,
            patches_per_tomo: int = 100,
            augment: bool = True,
    ):
        self.data_root = data_root
        self.P = patch_size
        self.positive_ratio = positive_ratio
        self.sigma_px = sigma_px
        self.ref_spacing = ref_spacing
        self.scale_sigma = scale_sigma
        self.patches_per_tomo = patches_per_tomo
        self.augment = augment

        # ── Build per-tomogram metadata ──────────────────────────────────────
        self.tomo_info: dict = {}
        for tid in tomo_ids:
            rows = df[df["tomo_id"] == tid]
            if rows.empty:
                continue

            # Robustly find the shape column (competition CSVs vary slightly)
            def _find_col(required_parts):
                for c in df.columns:
                    lc = c.lower()
                    if all(part.lower() in lc for part in required_parts):
                        return c
                raise KeyError(f"Could not find column containing: {required_parts}")

            shape_cols = [
                _find_col(["shape", "axis 0"]),
                _find_col(["shape", "axis 1"]),
                _find_col(["shape", "axis 2"]),
            ]

            spacing_col = next(
                (c for c in df.columns if "spacing" in c.lower() or "voxel" in c.lower()),
                None,
            )
            if spacing_col is None:
                raise KeyError("Could not find voxel spacing column")

            n_motors_col = next(
                (c for c in df.columns if "motor" in c.lower() and "number" in c.lower()),
                "Number of motors",
            )

            first_row = rows.iloc[0]

            shape = tuple(int(first_row[c]) for c in shape_cols)  # (D, H, W)
            voxel_spacing = float(first_row[spacing_col])

            motors = []
            for _, row in rows.iterrows():
                if int(row[n_motors_col]) > 0:
                    motors.append(
                        (
                            int(row["Motor axis 0"]),  # z  (slice index)
                            int(row["Motor axis 1"]),  # y
                            int(row["Motor axis 2"]),  # x
                        )
                    )

            self.tomo_info[tid] = {
                "motors": motors,
                "shape": shape,
                "voxel_spacing": voxel_spacing,
            }

        self.tomo_ids = list(self.tomo_info.keys())
        self._cache: dict = {}  # tomo_id → normalised float32 volume

    # ── Volume loading ───────────────────────────────────────────────────────

    def _load_volume(self, tomo_id: str) -> np.ndarray:
        """Load and normalise all slices; result is cached in this worker."""
        if tomo_id in self._cache:
            return self._cache[tomo_id]

        info = self.tomo_info[tomo_id]
        D, H, W = info["shape"]
        tomo_dir = os.path.join(self.data_root, tomo_id)

        volume = np.empty((D, H, W), dtype=np.float32)
        for i in range(D):
            # Files are 1-indexed: slice_0001.jpg … slice_NNNN.jpg
            path = os.path.join(tomo_dir, f"slice_{i:04d}.jpg")
            volume[i] = np.array(Image.open(path), dtype=np.float32)

        # Percentile normalisation — robust to cryo-EM intensity outliers
        p_lo = np.percentile(volume, 0.5)
        p_hi = np.percentile(volume, 99.5)
        volume = np.clip(volume, p_lo, p_hi)
        volume = (volume - p_lo) / (p_hi - p_lo + 1e-8)

        self._cache[tomo_id] = volume
        return volume

    # ── Heatmap construction ─────────────────────────────────────────────────

    def _effective_sigma(self, tomo_id: str) -> float:
        if self.scale_sigma:
            spacing = self.tomo_info[tomo_id]["voxel_spacing"]
            return self.sigma_px * (self.ref_spacing / spacing)
        return self.sigma_px

    @staticmethod
    def _gaussian_3d(
            patch_shape: tuple, ctr_zyx: tuple, sigma: float
    ) -> np.ndarray:
        D, H, W = patch_shape
        cz, cy, cx = ctr_zyx
        z = np.arange(D, dtype=np.float32)
        y = np.arange(H, dtype=np.float32)
        x = np.arange(W, dtype=np.float32)
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        return np.exp(
            -((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)
        ).astype(np.float32)

    # ── Patch sampling ───────────────────────────────────────────────────────

    def _sample_positive(self, tomo_id: str):
        """
        Returns (z0, y0, x0) patch origin + the selected motor's world coords.
        The motor is randomly jittered within ±P/4 of the patch centre so the
        model sees the motor at different positions within the field of view.
        """
        info = self.tomo_info[tomo_id]
        D, H, W = info["shape"]
        P = self.P

        # Pick one motor at random (usually only 1 per tomo)
        mz, my, mx = info["motors"][np.random.randint(len(info["motors"]))]

        jitter = P // 4  # ±32 for P=128

        def _clamp(v, lo, hi):
            return int(np.clip(v, lo, hi))

        z0 = _clamp(mz - P // 2 + np.random.randint(-jitter, jitter + 1), 0, D - P)
        y0 = _clamp(my - P // 2 + np.random.randint(-jitter, jitter + 1), 0, H - P)
        x0 = _clamp(mx - P // 2 + np.random.randint(-jitter, jitter + 1), 0, W - P)
        return (z0, y0, x0), (mz, my, mx)

    def _sample_negative(self, tomo_id: str, min_dist: int = 64):
        """
        Returns (z0, y0, x0) patch origin at least `min_dist` voxels from
        every motor in this tomogram (measured from patch centre to motor).
        Falls back after 200 failed attempts (extremely rare).
        """
        info = self.tomo_info[tomo_id]
        D, H, W = info["shape"]
        P = self.P
        motors = np.array(info["motors"]) if info["motors"] else None

        for _ in range(200):
            z0 = np.random.randint(0, D - P)
            y0 = np.random.randint(0, H - P)
            x0 = np.random.randint(0, W - P)
            if motors is None:
                return (z0, y0, x0)
            centre = np.array([z0 + P // 2, y0 + P // 2, x0 + P // 2])
            if np.min(np.linalg.norm(motors - centre, axis=1)) >= min_dist:
                return (z0, y0, x0)

        # Fallback — acceptable: labelling this patch as negative with a
        # near-by motor is slightly noisy but very rare.
        return (z0, y0, x0)

    # ── Augmentation ─────────────────────────────────────────────────────────

    @staticmethod
    def _augment(patch: torch.Tensor, hmap: torch.Tensor):
        """
        Label-preserving augmentations.
        Tensors are (C, D, H, W); spatial dims are 1, 2, 3.
        """
        # Random axis flips
        for dim in [1, 2, 3]:
            if np.random.random() < 0.5:
                patch = torch.flip(patch, [dim])
                hmap = torch.flip(hmap, [dim])

        # Random 90° rotation in the xy plane (dims 2, 3)
        k = np.random.randint(4)
        if k:
            patch = torch.rot90(patch, k, [2, 3])
            hmap = torch.rot90(hmap, k, [2, 3])

        return patch, hmap

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.tomo_ids) * self.patches_per_tomo

    def __getitem__(self, idx: int):
        """
        idx deterministically selects a tomogram (round-robin) but all patch
        positions and positive/negative decisions are stochastic, which is the
        correct behaviour for a patch-based 3-D dataset.
        """
        tomo_id = self.tomo_ids[idx % len(self.tomo_ids)]
        info = self.tomo_info[tomo_id]
        volume = self._load_volume(tomo_id)
        P = self.P

        has_motor = len(info["motors"]) > 0
        is_positive = has_motor and (np.random.random() < self.positive_ratio)

        if is_positive:
            (z0, y0, x0), (mz, my, mx) = self._sample_positive(tomo_id)
            patch = volume[z0: z0 + P, y0: y0 + P, x0: x0 + P].copy()
            sigma = self._effective_sigma(tomo_id)
            ctr_local = (mz - z0, my - y0, mx - x0)
            heatmap = self._gaussian_3d((P, P, P), ctr_local, sigma)
        else:
            z0, y0, x0 = self._sample_negative(tomo_id)
            patch = volume[z0: z0 + P, y0: y0 + P, x0: x0 + P].copy()
            heatmap = np.zeros((P, P, P), dtype=np.float32)

        patch_t = torch.from_numpy(patch).unsqueeze(0)  # (1, D, H, W)
        heatmap_t = torch.from_numpy(heatmap).unsqueeze(0)  # (1, D, H, W)

        if self.augment:
            patch_t, heatmap_t = self._augment(patch_t, heatmap_t)

        return patch_t, heatmap_t
