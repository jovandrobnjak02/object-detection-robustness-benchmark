import torch
import torch.nn as nn


S = 7   # grid size
B = 2   # boxes per cell
C = 3   # classes: Car/Van, Pedestrian, Cyclist


def _conv(in_ch, out_ch, k, s=1, p=None, use_batchnorm=False):
    if p is None:
        p = k // 2
    layers = [nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=not use_batchnorm)]
    if use_batchnorm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.1, inplace=True))
    return nn.Sequential(*layers)


class _BottleneckBlock(nn.Module):

    def __init__(self, ch_in, ch_mid, use_batchnorm: bool, use_residual: bool):
        super().__init__()
        self.conv1 = _conv(ch_in,  ch_mid, 1, use_batchnorm=use_batchnorm)
        self.conv2 = _conv(ch_mid, ch_in,  3, use_batchnorm=use_batchnorm)
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        if self.use_residual:
            out = out + x
        return out


class CustomCNN(nn.Module):
    def __init__(
        self,
        S: int = S,
        B: int = B,
        C: int = C,
        use_batchnorm: bool = False,
        use_residual: bool = False,
    ):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C

        bn = use_batchnorm
        res = use_residual

        self.stem = nn.Sequential(
            # Block 1
            _conv(3, 64, 7, s=2, use_batchnorm=bn),    # 448 → 224
            nn.MaxPool2d(2, 2),                          # 224 → 112

            # Block 2
            _conv(64, 192, 3, use_batchnorm=bn),
            nn.MaxPool2d(2, 2),                          # 112 → 56

            # Block 3
            _conv(192, 128, 1, use_batchnorm=bn),
            _conv(128, 256, 3, use_batchnorm=bn),
            _conv(256, 256, 1, use_batchnorm=bn),
            _conv(256, 512, 3, use_batchnorm=bn),
            nn.MaxPool2d(2, 2),                          # 56 → 28
        )

        # Block 4 — 4× bottleneck (residual applied per pair, ch_in=512)
        self.block4 = nn.Sequential(
            _BottleneckBlock(512, 256, bn, res),
            _BottleneckBlock(512, 256, bn, res),
            _BottleneckBlock(512, 256, bn, res),
            _BottleneckBlock(512, 256, bn, res),
            _conv(512, 512,  1, use_batchnorm=bn),
            _conv(512, 1024, 3, use_batchnorm=bn),
            nn.MaxPool2d(2, 2),                          # 28 → 14
        )

        # Block 5 — 2× bottleneck (residual applied per pair, ch_in=1024)
        self.block5 = nn.Sequential(
            _BottleneckBlock(1024, 512, bn, res),
            _BottleneckBlock(1024, 512, bn, res),
            _conv(1024, 1024, 3, use_batchnorm=bn),
            _conv(1024, 1024, 3, s=2, use_batchnorm=bn),  # 14 → 7
        )

        # Block 6
        self.block6 = nn.Sequential(
            _conv(1024, 1024, 3, use_batchnorm=bn),
            _conv(1024, 1024, 3, use_batchnorm=bn),
        )

        # Head — 1×1 conv reduces channels before flattening to preserve spatial info
        self.head = nn.Sequential(
            _conv(1024, 256, 1, use_batchnorm=bn),  # 1024 → 256 channels
            nn.Flatten(),                             # 256 * 7 * 7 = 12544
            nn.Linear(256 * S * S, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1 if use_batchnorm else 0.5),
            nn.Linear(4096, S * S * (B * 5 + C)),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.head(x)
        out = x.view(-1, self.S, self.S, self.B * 5 + self.C)

        # Apply sigmoid to cx, cy, conf, class — leave w, h raw
        # Indices to sigmoid: [0,1,4, 5,6,9, 10,11,12] for B=2, C=3
        sig_idx = []
        for i in range(self.B):
            base = i * 5
            sig_idx.extend([base, base + 1, base + 4])  # cx, cy, conf
        sig_idx.extend(range(self.B * 5, self.B * 5 + self.C))  # classes
        out[..., sig_idx] = torch.sigmoid(out[..., sig_idx])

        return out
