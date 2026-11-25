"""IR-ResNet backbone (the ResNet variant used by ArcFace): 112x112 aligned face -> 512-d embedding."""
import torch.nn as nn


class IBasicBlock(nn.Module):
    """Improved residual block: BN-conv-BN-PReLU-conv-BN (pre-activation style)."""

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        return out + identity


class IResNet(nn.Module):
    def __init__(self, layers, emb_size=512, dropout=0.4):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(64, layers[0], stride=2)    # 112 -> 56
        self.layer2 = self._make_layer(128, layers[1], stride=2)   # 56 -> 28
        self.layer3 = self._make_layer(256, layers[2], stride=2)   # 28 -> 14
        self.layer4 = self._make_layer(512, layers[3], stride=2)   # 14 -> 7
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * 7 * 7, emb_size)                 # fixes input size at 112x112
        self.features = nn.BatchNorm1d(emb_size, eps=1e-5)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, planes, blocks, stride):
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
            nn.BatchNorm2d(planes, eps=1e-5),
        )
        layers = [IBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers += [IBasicBlock(planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.bn2(x)
        x = self.dropout(x.flatten(1))
        return self.features(self.fc(x))


ARCHS = {
    "r18": [2, 2, 2, 2],
    "r34": [3, 4, 6, 3],
    "r50": [3, 4, 14, 3],
}


def build_backbone(arch="r34", emb_size=512, dropout=0.4):
    return IResNet(ARCHS[arch], emb_size=emb_size, dropout=dropout)
