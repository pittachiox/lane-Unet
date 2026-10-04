"""Custom lane-segmentation U-Net, trained from scratch.
Input : (B, 3, 48, 48) RGB in [0, 1]
Output: (B, 1, 48, 48) logits (sigmoid > 0.5 -> binary lane mask)
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """[Conv3x3 -> BN -> ReLU] x 2"""

    def __init__(self, cin, cout):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class LaneUNet(nn.Module):
    """3-level U-Net: 48 -> 24 -> 12 -> 6 (bottleneck). 48 divides by 2 three times, so skips align exactly."""

    def __init__(self, base=16):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.enc1 = ConvBlock(3, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.bott = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = ConvBlock(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = ConvBlock(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = ConvBlock(c1 * 2, c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)                 # 16 x 48 x 48
        e2 = self.enc2(self.pool(e1))     # 32 x 24 x 24
        e3 = self.enc3(self.pool(e2))     # 64 x 12 x 12
        b = self.bott(self.pool(e3))      # 128 x 6 x 6
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


if __name__ == "__main__":
    m = LaneUNet()
    print(m(torch.randn(2, 3, 48, 48)).shape, sum(p.numel() for p in m.parameters()), "params")
