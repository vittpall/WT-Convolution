"""ResNet-20 (He et al. CIFAR layout) with the second 3x3 conv of every block
optionally replaced by a TransformConv2D. Matches the per-directory models in
CIFAR10/resnet20*, but with the transform, paths, residual flag and Haar levels
as explicit config instead of separate copies."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import TransformConv2D


class ResidualBlock(nn.Module):
    def __init__(self, h, w, cin, cout, stride, cfg):
        super().__init__()
        if cfg['kind'] == 'conv':
            second = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        else:
            second = TransformConv2D(h, w, cout, cout, pods=cfg['pods'],
                                     residual=cfg['residual'], kind=cfg['kind'],
                                     levels=cfg.get('levels'))
        self.left = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            second,
            nn.BatchNorm2d(cout),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return F.relu(self.left(x) + self.shortcut(x))


class ResNet20(nn.Module):
    def __init__(self, cfg, num_classes=10, size=32):
        super().__init__()
        self.cin, self.h, self.w = 16, size, size
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.layer1 = self._stage(16, 1, cfg)
        self.layer2 = self._stage(32, 2, cfg)
        self.layer3 = self._stage(64, 2, cfg)
        self.fc = nn.Linear(64, num_classes)

    def _stage(self, cout, stride, cfg):
        blocks = []
        for s in [stride, 1, 1]:
            self.h //= s
            self.w //= s
            blocks.append(ResidualBlock(self.h, self.w, self.cin, cout, s, cfg))
            self.cin = cout
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.layer3(self.layer2(self.layer1(self.conv1(x))))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


# Residual convention of Pan et al. (README): 1-path residual=True, multi-path residual=False.
def _t(kind, p, residual=None, levels=None):
    return dict(kind=kind, pods=p, residual=(p == 1) if residual is None else residual, levels=levels)


CONFIGS = {
    'resnet20': dict(kind='conv'),
    # E1: matched comparison, both orthonormal, same residual convention
    **{f'ht_p{p}': _t('ht', p) for p in (1, 2, 3)},
    **{f'haar_p{p}': _t('haar', p) for p in (1, 2, 3)},
    # published HT baseline scaling (Pan et al. CIFAR code)
    'htpan_p3': _t('ht_pan', 3),
    # setting used for the paper's Haar 3-path number (residual=True), both transforms
    'ht_p3_res': _t('ht', 3, residual=True),
    'haar_p3_res': _t('haar', 3, residual=True),
    # N1: learnable orthogonal wavelet (Haar init), angle per level / per butterfly
    'lhaar_p3': _t('lhaar', 3),
    'lhaar_p3_res': _t('lhaar', 3, residual=True),
    'lhaarpos_p3': _t('lhaar_pos', 3),
    'lhaarpos_p3_res': _t('lhaar_pos', 3, residual=True),
    # E3: Haar decomposition levels (full = 5/4/3 levels at 32/16/8 px)
    **{f'haar_p3_L{L}': _t('haar', 3, levels=L) for L in (1, 2, 3)},
    **{f'ht_p3_L{L}': _t('ht', 3, levels=L) for L in (1, 2, 3)},  # block-HT control
}


def build(name):
    return ResNet20(CONFIGS[name])
