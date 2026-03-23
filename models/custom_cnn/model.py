import torch
import torch.nn as nn


S = 14  # grid size
B = 2   # boxes per cell
C = 10  # number of classes


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
            _conv(3, 32, 7, s=2, use_batchnorm=bn),     # 448 to 224
            nn.MaxPool2d(2, 2),                           # 224 to 112

            # Block 2
            _conv(32, 96, 3, use_batchnorm=bn),
            nn.MaxPool2d(2, 2),                           # 112 to 56

            # Block 3
            _conv(96, 128, 1, use_batchnorm=bn),
            _conv(128, 256, 3, use_batchnorm=bn),
            nn.MaxPool2d(2, 2),                           # 56 to 28
        )

        # Block 4 - 2× bottleneck + maxpool
        self.block4 = nn.Sequential(
            _BottleneckBlock(256, 128, bn, res),
            _BottleneckBlock(256, 128, bn, res),
            _conv(256, 256, 3, use_batchnorm=bn),
            nn.MaxPool2d(2, 2),                           # 28 to 14
        )

        # Block 5 - 2× bottleneck + convs
        self.block5 = nn.Sequential(
            _BottleneckBlock(256, 128, bn, res),
            _BottleneckBlock(256, 128, bn, res),
            _conv(256, 256, 3, use_batchnorm=bn),
            _conv(256, 256, 3, use_batchnorm=bn),          # 14 to 14
        )

        # Head - fully convolutional (no FC layers)
        out_ch = B * 5 + C
        self.head = nn.Sequential(
            _conv(256, 128, 1, use_batchnorm=bn),
            nn.Conv2d(128, out_ch, 1),  # raw output, no activation
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.head(x)                          # (batch, B*5+C, S, S)
        out = x.permute(0, 2, 3, 1).contiguous()  # (batch, S, S, B*5+C)

        # Apply sigmoid to cx, cy, conf - leave w, h raw, classes as logits (BCE with logits in loss)
        sig_idx = []
        for i in range(self.B):
            base = i * 5
            sig_idx.extend([base, base + 1, base + 4])  # cx, cy, conf
        out[..., sig_idx] = torch.sigmoid(out[..., sig_idx])

        return out
