import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import os
import math

from timm.models.layers import trunc_normal_
# from model.blocks import CBlock_ln, SwinTransformerBlock
from model.restormer_arch import TransformerBlock
from model.CalibrateLayer import DRConv2d
from model.WaveletTransform import WaveletTransform

# Short Cut Connection on Final Layer
class Local(nn.Module):
    def __init__(self, in_dim=3, dim=16, number=1):
        super(Local, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2d(in_dim, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        block_t = TransformerBlock(dim=dim, num_heads=4, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias') # head number
        
        blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
        
        self.block = TransformerBlock(dim=dim*2, num_heads=8, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias') # head number
        self.conv_block = nn.Conv2d(dim, dim * 2, 3, 1, 1)
       
    
        # Calibrate Branch
        self.drconv2d = DRConv2d(48, 48, kernel_size=3, region_num=5, stride=1, padding=1)
        self.wavelet_dec = WaveletTransform(scale=2, dec=True)    ## 小波变换
        self.wavelet_rec = WaveletTransform(scale=2, dec=False)   ## 小波逆变换
        
        self.enhance = EnhanceNetwork(layers=1, channels=48)
        
        self.mul_blocks = nn.Sequential(*blocks1)
        self.add_blocks = nn.Sequential(*blocks2)

        self.end = nn.Sequential(nn.Conv2d(dim * 3, 3, 3, 1, 1), nn.ReLU())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
            
            

    def forward(self, img):
        img1 = self.relu(self.conv1(img))
        # short cut connection
        img1 = self.add_blocks(img1).add(img1)
        img1 = self.mul_blocks(img1).mul(img1)
        
        # CalibrateLayer Branch
        w_img = self.wavelet_dec(img1)
        w_img = self.drconv2d(w_img)       
        w_img = self.wavelet_rec(w_img)
        
        img1 = self.conv_block(img1)
        img1 = self.block(img1)
        
        img1 = torch.cat([img1, w_img], dim=1)
        
        img1 = self.enhance(img1)
        end = self.end(img1)
        return  end
#         return mul, add

class Net(nn.Module):
    def __init__(self, in_dim=3):
        super(Net, self).__init__()
        #self.local_net = Local_pred()
        self.local_net = Local(in_dim=in_dim)

    def apply_color(self, image, ccm):
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)

    def forward(self, img_low):
                
        img_high = self.local_net(img_low)
        
        _ = ''
        return _, _, img_high


#         return mul, add, img_high
        

class EnhanceNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(EnhanceNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels*2, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels*2, out_channels=channels*2, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels*2),
            nn.ReLU()
        )
        self.sft = SFT(channels*2)
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)
            self.blocks.append(self.sft)



        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels*2, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)
        illu = fea + input
        return illu


class SFT(nn.Module):
    def __init__(self, dim):
        super(SFT, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(dim, dim * 2, 1)
        self.SFT_scale_conv1 = nn.Conv2d(dim * 2, dim, 1)
        self.SFT_shift_conv0 = nn.Conv2d(dim, dim * 2, 1)
        self.SFT_shift_conv1 = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x):
        
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x), 0.1, inplace=True))
        return x * scale + shift


