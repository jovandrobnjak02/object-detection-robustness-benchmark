import math
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork


B = 2   # boxes per cell
C = 10  # number of classes
FPN_OUT = 256


class CustomCNN(nn.Module):
    def __init__(self, B: int = B, C: int = C):
        super().__init__()
        self.B = B
        self.C = C

        # Pretrained ResNet-50 backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # 256ch
        self.layer2 = resnet.layer2  # 512ch
        self.layer3 = resnet.layer3  # 1024ch
        self.layer4 = resnet.layer4  # 2048ch

        # FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[512, 1024, 2048],
            out_channels=FPN_OUT,
        )

        # Scale-specific decoupled heads (YOLOX-style: separate cls and reg branches)
        self.cls_heads = nn.ModuleList([self._make_cls_head() for _ in range(3)])
        self.reg_heads = nn.ModuleList([self._make_reg_head() for _ in range(3)])

        # Sigmoid indices: cx, cy, conf for each box predictor
        self._sig_idx = []
        for i in range(B):
            base = i * 5
            self._sig_idx.extend([base, base + 1, base + 4])

        self._init_head_weights()

    def _make_cls_head(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(FPN_OUT, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.C, 1),
        )

    def _make_reg_head(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(FPN_OUT, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.B * 5, 1),
        )

    def _init_head_weights(self):
        for head_list in (self.cls_heads, self.reg_heads):
            for head in head_list:
                for m in head.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                                nonlinearity="relu")
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)
        # Focal loss prior: start cls outputs at low confidence (pi=0.01)
        pi = 0.01
        with torch.no_grad():
            for cls_head in self.cls_heads:
                cls_head[-1].bias.fill_(-math.log((1 - pi) / pi))

    def freeze_backbone(self):
        for mod in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in mod.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = True

    def _apply_head(self, feat: torch.Tensor, scale_idx: int) -> torch.Tensor:
        reg = self.reg_heads[scale_idx](feat)  # (N, B*5, S, S)
        cls = self.cls_heads[scale_idx](feat)  # (N, C, S, S)
        # Concatenate reg then cls to preserve output format [B*5 | C]
        out = torch.cat([reg, cls], dim=1).permute(0, 2, 3, 1).contiguous()  # (N, S, S, B*5+C)
        out[..., self._sig_idx] = torch.sigmoid(out[..., self._sig_idx])
        return out

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x  = self.stem(x)
        x  = self.layer1(x)
        c3 = self.layer2(x)   # 512ch
        c4 = self.layer3(c3)  # 1024ch
        c5 = self.layer4(c4)  # 2048ch

        fpn_out = self.fpn({"c3": c3, "c4": c4, "c5": c5})

        return [
            self._apply_head(fpn_out["c3"], 0),  # P3: 80×80 at 640px
            self._apply_head(fpn_out["c4"], 1),  # P4: 40×40
            self._apply_head(fpn_out["c5"], 2),  # P5: 20×20
        ]
