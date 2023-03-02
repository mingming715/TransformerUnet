#!/usr/bin/python

# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class AttentionGate(nn.Module):
    def __init__(self, in_channel, gating_channel, inter_channel):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv2d(gating_channel, inter_channel, kernel_size=1)
        self.W_x = nn.Conv2d(in_channel, inter_channel, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.psi = nn.Conv2d(inter_channel, 1, kernel_size=1)
        self.sig = nn.Sigmoid()
        
    def forward(self, x, g):
        g_conv = self.W_g(g)
        x_conv = self.W_x(x)
        out = self.relu(g_conv + x_conv)
        out = self.sig(self.psi(out))
        out = F.upsample(out, size=x.size()[2:], mode='bilinear')
        out = x * out
        return out

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)
        self.conv_x2 = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # x2 skip부분
        # 추가한거임
        # x2 = self.conv_x2(x2)
        x = torch.cat([x2, x1], dim=1)

        x_final = self.conv(x)
        return x_final

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x    