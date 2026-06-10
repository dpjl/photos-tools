"""bopbtl_networks.py — Réseau de détection de rayures (U-Net).

Code adapté du projet **Microsoft "Bringing Old Photos Back to Life"**
(CVPR 2020 oral), fichiers ``Global/detection_models/networks.py`` et
``antialiasing.py``.  Licence MIT — Copyright (c) Microsoft Corporation.
https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life

Version allégée pour l'inférence seule :
  · seule la classe ``UNet`` du détecteur est conservée (pas de pix2pix /
    générateurs CycleGAN inutilisés) ;
  · ``sync_bn`` est retiré (l'original le rendait inopérant à l'inférence —
    réassigner ``self`` dans ``__init__`` n'a aucun effet sur l'objet construit) ;
  · la BatchNorm standard ``nn.BatchNorm2d`` charge directement les poids
    ``FT_Epoch_latest.pt`` (clés sans préfixe ``module.``).

Le réseau prend une image **niveaux de gris** normalisée et renvoie une carte
de logits ; appliquer ``torch.sigmoid`` pour obtenir la probabilité de rayure.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Anti-crénelage (BlurPool, d'après adobe/antialiased-cnns) ─────────────────

class Downsample(nn.Module):
    """Sous-échantillonnage anti-crénelé par filtre binomial puis stride."""

    def __init__(self, channels: int, filt_size: int = 3, stride: int = 2) -> None:
        super().__init__()
        self.stride = stride
        pad = [(filt_size - 1) // 2, int(np.ceil((filt_size - 1) / 2))] * 2
        self.pad = nn.ReflectionPad2d(pad)
        a = {
            1: [1.0],
            2: [1.0, 1.0],
            3: [1.0, 2.0, 1.0],
            4: [1.0, 3.0, 3.0, 1.0],
            5: [1.0, 4.0, 6.0, 4.0, 1.0],
            6: [1.0, 5.0, 10.0, 10.0, 5.0, 1.0],
            7: [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0],
        }[filt_size]
        a = np.array(a)
        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / filt.sum()
        self.register_buffer("filt", filt[None, None].repeat((channels, 1, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(self.pad(x), self.filt, stride=self.stride, groups=x.shape[1])


# ── Blocs U-Net ───────────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    def __init__(self, conv_num, in_size, out_size, padding, batch_norm):
        super().__init__()
        block = []
        for _ in range(conv_num):
            block += [
                nn.ReflectionPad2d(int(padding)),
                nn.Conv2d(in_size, out_size, kernel_size=3, padding=0),
            ]
            if batch_norm:
                block.append(nn.BatchNorm2d(out_size))
            block.append(nn.LeakyReLU(0.2, True))
            in_size = out_size
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class _UpBlock(nn.Module):
    def __init__(self, conv_num, in_size, out_size, padding, batch_norm):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=0),
        )
        self.conv_block = _ConvBlock(conv_num, in_size, out_size, padding, batch_norm)

    @staticmethod
    def _center_crop(layer, target):
        _, _, h, w = layer.size()
        dy = (h - target[0]) // 2
        dx = (w - target[1]) // 2
        return layer[:, :, dy:dy + target[0], dx:dx + target[1]]

    def forward(self, x, bridge):
        up = self.up(x)
        bridge = self._center_crop(bridge, up.shape[2:])
        return self.conv_block(torch.cat([up, bridge], 1))


# ── Détecteur ─────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """U-Net de détection de rayures (configuration des poids officiels :
    ``depth=4, conv_num=2, wf=6``, entrée/sortie 1 canal)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        depth: int = 4,
        conv_num: int = 2,
        wf: int = 6,
        padding: bool = True,
        batch_norm: bool = True,
        antialiasing: bool = True,
    ) -> None:
        super().__init__()
        self.first = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 2 ** wf, kernel_size=7),
            nn.LeakyReLU(0.2, True),
        )
        prev = 2 ** wf

        self.down_path = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        for i in range(depth):
            if antialiasing:
                self.down_sample.append(nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(prev, prev, kernel_size=3, stride=1, padding=0),
                    nn.BatchNorm2d(prev),
                    nn.LeakyReLU(0.2, True),
                    Downsample(channels=prev, stride=2),
                ))
            else:
                self.down_sample.append(nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(prev, prev, kernel_size=4, stride=2, padding=0),
                    nn.BatchNorm2d(prev),
                    nn.LeakyReLU(0.2, True),
                ))
            self.down_path.append(_ConvBlock(conv_num, prev, 2 ** (wf + i + 1), padding, batch_norm))
            prev = 2 ** (wf + i + 1)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up_path.append(_UpBlock(conv_num, prev, 2 ** (wf + i), padding, batch_norm))
            prev = 2 ** (wf + i)

        self.last = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(prev, out_channels, kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first(x)
        blocks = []
        for i, down_block in enumerate(self.down_path):
            blocks.append(x)
            x = self.down_sample[i](x)
            x = down_block(x)
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        return self.last(x)
