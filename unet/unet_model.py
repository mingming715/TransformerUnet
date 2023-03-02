#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F

# python 3 confusing imports :(
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

        self.ag1 = AttentionGate(512, 512, 512)
        self.ag2 = AttentionGate(256, 256, 256)
        self.ag3 = AttentionGate(128, 128, 128)
        self.ag4 = AttentionGate(64, 64, 64)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        ag1 = self.ag1(x4, x5)
        up1 = self.up1(x5, ag1)
        ag2 = self.ag2(x3, up1)
        up2 = self.up2(up1, ag2)
        ag3 = self.ag3(x2, up2)
        up3 = self.up3(up2, ag3)
        ag4 = self.ag4(x1, up3)
        x = self.up4(up3, ag4)
        x = self.outc(x).sigmoid()
        return x
