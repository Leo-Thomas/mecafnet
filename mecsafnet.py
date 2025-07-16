"""
MeCSAFNet - Multi-encoder ConvNeXt Network with Smooth Attentional Feature Fusion for Multispectral Semantic Segmentation

Author: Leo Thomas Ramos
License: GNU General Public License v3.0

Requirements:
    - PyTorch
    - TorchVision

Model Components:
    - ASAU: Custom activation
    - DecoderBlock: Convolution + PixelShuffle upsampling
    - CBAM: Convolutional Block Attention Module
    - FPN_fuse: Feature Pyramid Network with smooth attention and fusion
    - rgb_net / nir_net: Dual-branch encoders
    - MeCSAFNet: Main model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_base, convnext_tiny, convnext_small, convnext_large

class ASAU(nn.Module):
    def __init__(self):
        super(ASAU, self).__init__()
        self.w0 = nn.Parameter(torch.tensor(0.05))
        self.w1 = nn.Parameter(torch.tensor(0.5))
        self.w2 = nn.Parameter(torch.tensor(1.5))

    def forward(self, x):
        return self.w0 * x + ((1.0 - self.w0) * x * torch.tanh(self.w2 * F.softplus((1.0 - self.w0) * self.w1 * x)))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True),
            ASAU(),
            nn.PixelShuffle(upscale_factor=2)
        )

    def forward(self, x):
        return self.decode(x)


def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y


class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, dilation=1, bias=bias)

    def forward(self, x):
        max_out = torch.max(x, 1)[0].unsqueeze(1)
        avg_out = torch.mean(x, 1).unsqueeze(1)
        concat = torch.cat((max_out, avg_out), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output) * x
        return output


class CAM(nn.Module):
    def __init__(self, channels, r=16):
        super(CAM, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(channels, channels // r),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels)
        )

    def forward(self, x):
        max_out = F.adaptive_max_pool2d(x, 1)
        avg_out = F.adaptive_avg_pool2d(x, 1)
        b, c, _, _ = x.size()
        max_proj = self.linear(max_out.view(b, c)).view(b, c, 1, 1)
        avg_proj = self.linear(avg_out.view(b, c)).view(b, c, 1, 1)
        out = F.sigmoid(max_proj + avg_proj) * x
        return out


class CBAM(nn.Module):
    def __init__(self, channels, r=16):
        super(CBAM, self).__init__()
        self.cam = CAM(channels, r)
        self.sam = SAM()

    def forward(self, x):
        x = self.cam(x)
        x = self.sam(x)
        return x + x


class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[32, 64, 128, 256], fpn_out=32):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(c, fpn_out, kernel_size=1) for c in feature_channels[1:]])
        self.smooth_conv = nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1) for _ in range(len(feature_channels) - 1)])
        self.cbam_blocks = nn.ModuleList([CBAM(fpn_out) for _ in range(len(feature_channels) - 1)])
        self.smooth_act = nn.ModuleList([ASAU() for _ in range(len(feature_channels) - 1)])
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels) * fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            #nn.ReLU(inplace=True),
            ASAU()
        )

    def forward(self, features):
        features[1:] = [conv(f) for f, conv in zip(features[1:], self.conv1x1)]
        P = []
        for i in reversed(range(1, len(features))):
            fused = up_and_add(features[i], features[i - 1])
            smoothed = self.smooth_conv[i - 1](fused)
            activated = self.smooth_act[i - 1](smoothed)
            recalibrated = self.cbam_blocks[i - 1](activated)
            P.append(recalibrated)
        P = list(reversed(P))
        P.append(features[-1])
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(p, size=(H, W), mode='bilinear', align_corners=True) for p in P[1:]]
        x = self.conv_fusion(torch.cat(P, dim=1))
        return x


class rgb_net(nn.Module):
    def __init__(self, num_classes, filters=32):
        super().__init__()
        self.rgb = convnext_base(pretrained=True) # <-- change convnext version here
        
        ## convnext_tiny and convnext_small
        # self.dec5 = DecoderBlock(768, filters * 16)
        # self.dec4 = DecoderBlock(768 + filters * 4, filters * 16)
        # self.dec3 = DecoderBlock(768 + filters * 4, filters * 8)
        # self.dec2 = DecoderBlock(384 + filters * 2, filters * 4)
        # self.dec1 = DecoderBlock(192 + filters * 1, filters * 2)
        
        ## convnext_base
        self.dec5 = DecoderBlock(1024, filters * 16)
        self.dec4 = DecoderBlock(1024 + filters * 4, filters * 16)
        self.dec3 = DecoderBlock(1024 + filters * 4, filters * 8)
        self.dec2 = DecoderBlock(512 + filters * 2, filters * 4)
        self.dec1 = DecoderBlock(256 + filters * 1, filters * 2)
        
        ## convnext_large
        # self.dec5 = DecoderBlock(1536, filters * 16)
        # self.dec4 = DecoderBlock(1536 + filters * 4, filters * 16)
        # self.dec3 = DecoderBlock(1536 + filters * 4, filters * 8)
        # self.dec2 = DecoderBlock(768 + filters * 2, filters * 4)
        # self.dec1 = DecoderBlock(384 + filters * 1, filters * 2)

    def forward(self, rgb):
        rgb0 = self.rgb.features[0](rgb)
        rgb1 = self.rgb.features[2](self.rgb.features[1](rgb0))
        rgb2 = self.rgb.features[4](self.rgb.features[3](rgb1))
        rgb3 = self.rgb.features[6](self.rgb.features[5](rgb2))
        rgb4 = self.rgb.features[7](rgb3)
        rgb4 = F.max_pool2d(rgb4, kernel_size=2, stride=2)

        dec5 = self.dec5(F.max_pool2d(rgb4, kernel_size=2, stride=2))
        dec4 = self.dec4(torch.cat((rgb4, dec5), dim=1))
        dec3 = self.dec3(torch.cat((rgb3, dec4), dim=1))
        dec2 = self.dec2(torch.cat((rgb2, dec3), dim=1))
        dec1 = self.dec1(torch.cat((rgb1, dec2), dim=1))

        return dec1, dec2, dec3, dec4


class nir_net(nn.Module):
    def __init__(self, num_classes, filters=32):
        super().__init__()
        self.nir = convnext_base(pretrained=True) # <-- change convnext version here
        
        ## convnext_tiny and convnext_small
        # self.dec5 = DecoderBlock(768, filters * 16)
        # self.dec4 = DecoderBlock(768 + filters * 4, filters * 16)
        # self.dec3 = DecoderBlock(768 + filters * 4, filters * 8)
        # self.dec2 = DecoderBlock(384 + filters * 2, filters * 4)
        # self.dec1 = DecoderBlock(192 + filters * 1, filters * 2)
        
        ## convnext_base
        self.dec5 = DecoderBlock(1024, filters * 16)
        self.dec4 = DecoderBlock(1024 + filters * 4, filters * 16)
        self.dec3 = DecoderBlock(1024 + filters * 4, filters * 8)
        self.dec2 = DecoderBlock(512 + filters * 2, filters * 4)
        self.dec1 = DecoderBlock(256 + filters * 1, filters * 2)
        
        ## convnext_large
        # self.dec5 = DecoderBlock(1536, filters * 16)
        # self.dec4 = DecoderBlock(1536 + filters * 4, filters * 16)
        # self.dec3 = DecoderBlock(1536 + filters * 4, filters * 8)
        # self.dec2 = DecoderBlock(768 + filters * 2, filters * 4)
        # self.dec1 = DecoderBlock(384 + filters * 1, filters * 2)

    def forward(self, nir):
        nir0 = self.nir.features[0](nir)
        nir1 = self.nir.features[2](self.nir.features[1](nir0))
        nir2 = self.nir.features[4](self.nir.features[3](nir1))
        nir3 = self.nir.features[6](self.nir.features[5](nir2))
        nir4 = self.nir.features[7](nir3)
        nir4 = F.max_pool2d(nir4, kernel_size=2, stride=2)

        dec5 = self.dec5(F.max_pool2d(nir4, kernel_size=2, stride=2))
        dec4 = self.dec4(torch.cat((nir4, dec5), dim=1))
        dec3 = self.dec3(torch.cat((nir3, dec4), dim=1))
        dec2 = self.dec2(torch.cat((nir2, dec3), dim=1))
        dec1 = self.dec1(torch.cat((nir1, dec2), dim=1))

        return dec1, dec2, dec3, dec4


class MeCSAFNet(nn.Module):
    def __init__(self, num_classes):
        super(MeCSAFNet, self).__init__()
        self.rgb = rgb_net(num_classes)
        self.nir = nir_net(num_classes)
        self.FPN = FPN_fuse([32, 64, 128, 256], 32)
        self.fuse = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)

    def forward(self, rgb_nir_ndvi_ndwi):
        input_size = rgb_nir_ndvi_ndwi.size()[2:]
        
        rgb_input = rgb_nir_ndvi_ndwi[:, :3]
        nir_input = rgb_nir_ndvi_ndwi[:, 3:6]

        rgb_feats = self.rgb(rgb_input)
        nir_feats = self.nir(nir_input)

        features = [torch.cat((r, n), dim=1) for r, n in zip(rgb_feats, nir_feats)]
        fpn = self.FPN(features)
        x = F.interpolate(fpn, size=input_size, mode="bicubic", align_corners=True)
        return self.fuse(x)

if __name__ == "__main__":

    num_classes = 6

    model = MeCSAFNet(num_classes=num_classes)

    dummy_input = torch.rand(1, 6, 512, 512)

    output = model(dummy_input)

    print(f"Output shape: {output.shape}")