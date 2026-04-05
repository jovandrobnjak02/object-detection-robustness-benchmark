import math
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork

FPN_OUT = 256

# Default boxes-per-cell must match decoder/loss (B=2)
class CustomCNN(nn.Module):
    def __init__(self, B: int = 2, C: int = 10):
        super().__init__()
        self.B = B
        self.C = C

        # Backbone: ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1 
        self.layer2 = resnet.layer2 
        self.layer3 = resnet.layer3 
        self.layer4 = resnet.layer4 

        # FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[512, 1024, 2048],
            out_channels=FPN_OUT,
        )

        # Heads
        self.cls_head = self._make_cls_head()
        self.reg_head = self._make_reg_head()
        self._init_head_weights()

    def _make_cls_head(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(128, self.C, 1), 
        )

    def _make_reg_head(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(128, self.B * 5, 1), 
        )

    def _init_head_weights(self):
        with torch.no_grad():
            for head in (self.cls_head, self.reg_head):
                for m in head.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu")
                        if m.bias is not None: nn.init.zeros_(m.bias)
                    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)
            
            # Focal loss prior initialization
            pi = 0.01
            self.cls_head[-1].bias.fill_(-math.log((1 - pi) / pi))

            nn.init.normal_(self.reg_head[-1].weight, mean=0.0, std=0.01)
            b = self.reg_head[-1].bias
            for j in range(self.B):
                base = j * 5
                b[base + 0:base + 4] = 0.0
                b[base + 4] = -4.0 # Initial low objectness

    def freeze_backbone(self):
        for mod in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in mod.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = True

    def unfreeze_layer4_only(self):
        """Unfreeze only layer4 (fine-grained feature learning) keep stem/layer1-3 frozen."""
        for p in self.layer4.parameters():
            p.requires_grad = True

    def unfreeze_progressive(self):
        """Unfreeze progressively: layer4 -> layer3 -> layer2 -> layer1 -> stem."""
        # Check if layer4 is already unfrozen; if so, unfreeze layer3, etc.
        layers = [self.layer4, self.layer3, self.layer2, self.layer1, self.stem]
        for layer in layers:
            if not any(p.requires_grad for p in layer.parameters()):
                for p in layer.parameters():
                    p.requires_grad = True
                break  # Unfreeze one layer at a time

    def _apply_head(self, feat: torch.Tensor) -> torch.Tensor:
        reg = self.reg_head(feat)  
        cls = self.cls_head(feat)  
        return torch.cat([reg, cls], dim=1).permute(0, 2, 3, 1).contiguous()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        fpn_out = self.fpn({"c3": c3, "c4": c4, "c5": c5})
        return [self._apply_head(fpn_out[k]) for k in ["c3", "c4", "c5"]]