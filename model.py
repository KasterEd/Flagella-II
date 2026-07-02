import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv3d → BN → ReLU) × 2  — standard U-Net building block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet3D(nn.Module):
    """
    Args:
        in_ch  : number of input channels (1 for single-channel tomogram patches).
        out_ch : number of output channels (1 for scalar heatmap).
        f      : base feature multiplier.
                 Feature counts: f, 2f, 4f, 8f, 16f across encoder levels.
                 f=32  → ~10 M parameters, safe at batch 4 on 64 GB A100.
                 f=64  → ~40 M parameters, still fits but tighter.
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 1, f: int = 32):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = DoubleConv(in_ch, f)  # → (B,  f, 128, 128, 128)
        self.enc2 = DoubleConv(f, f * 2)  # → (B, 2f,  64,  64,  64)
        self.enc3 = DoubleConv(f * 2, f * 4)  # → (B, 4f,  32,  32,  32)
        self.enc4 = DoubleConv(f * 4, f * 8)  # → (B, 8f,  16,  16,  16)

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = DoubleConv(f * 8, f * 16)  # → (B, 16f,  8,  8,  8)

        self.pool = nn.MaxPool3d(kernel_size=2)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up4 = nn.ConvTranspose3d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(f * 16, f * 8)

        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(f * 8, f * 4)

        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(f * 4, f * 2)

        self.up1 = nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(f * 2, f)

        # ── Output head ──────────────────────────────────────────────────────
        self.head = nn.Conv3d(f, out_ch, kernel_size=1)

        # Weight initialisation: Kaiming for Conv, 1/0 for BN
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)
