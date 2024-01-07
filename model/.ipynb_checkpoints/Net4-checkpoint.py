import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import os
import math

from torchvision import transforms
import cv2
import numpy as np
from timm.models.layers import trunc_normal_
from model.blocks import CBlock_ln, SwinTransformerBlock

# Short Cut Connection on Final Layer
class Local_pred_S(nn.Module):
    def __init__(self, in_dim=3, dim=16, number=1):
        super(Local_pred_S, self).__init__()
        # initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, dim, 3, padding=1, groups=1),
            nn.Conv2d(dim, dim, 3, padding=1)
        )
        
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        block_t = SwinTransformerBlock(dim)  # head number
        
        blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
        blocks1.append(FeedForward(dim=dim, ffn_expansion_factor=2.66, bias=True))
        blocks2.append(FeedForward(dim=dim, ffn_expansion_factor=2.66, bias=True))
        self.mul_blocks = nn.Sequential(*blocks1)
        self.add_blocks = nn.Sequential(*blocks2)

        self.mul_end = nn.Sequential(nn.Conv2d(dim, 3, 3, padding = 1), nn.ReLU())
        self.add_end = nn.Sequential(nn.Conv2d(dim, 3, 3, padding = 1), nn.Tanh())
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
        mul = self.mul_blocks(img1) + img1
        add = self.add_blocks(img1) + img1
        mul = self.mul_end(mul)
        add = self.add_end(add)
        return mul, add

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.relu(x2)*x1 + F.relu(x1)*x2
        x = self.project_out(x)
        return x    

class Net4(nn.Module):
    def __init__(self, in_dim=3):
        super(Net4, self).__init__()
        #self.local_net = Local_pred()
        self.local_net = Local_pred_S(in_dim=in_dim, dim=32)
        self.fft_block = FFT_Block()
        
    def apply_color(self, image, ccm):
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)

    def forward(self, img_low):
        #print(self.with_global)
        mul, add = self.local_net(img_low)
        img_high = (img_low.mul(mul)).add(add)
        
        
        fft_map = self.fft_block(img_low)
        img_high = img_high + fft_map
        
        return mul, add, img_high
        
class global_module(nn.Module):
    def __init__(self, inplanes, act=nn.LeakyReLU(negative_slope=0.2,inplace=True), bias=False):
        super(global_module, self).__init__()

        self.conv_mask = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=bias)
        self.linear = nn.Linear(inplanes, 1)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        input_x = input_x.view(batch, channel, height * width)
        input_x = input_x.unsqueeze(1)
        
        context_mask = self.conv_mask(x)
        context_mask = context_mask.view(batch, -1, channel)
        context_mask = self.linear(context_mask)
        context_mask = context_mask.unsqueeze(1)
        context = torch.matmul(input_x, context_mask)
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        x = x + context
        return x
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return out + x


class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_embedding, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        x = self.proj(x)
        return x
    
class CA(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CA, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x, fft_map=None):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
    
    

def fft_img(img):
    
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)

    fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg) 
    
    return transforms.ToTensor()(iimg)


class FFT_Block(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, num_heads=4, type='exp'):
        super(FFT_Block, self).__init__()
        # main blocks
        self.conv_large = conv_embedding(in_channels, out_channels)
        self.block_t = SwinTransformerBlock(out_channels)
        self.generator = CA(in_dim=out_channels)
        self.conv_fft = nn.Conv2d(3, 3, kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.conv_A = nn.Conv2d(out_channels, 3, kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.gm = ChannelAttention(out_channels)


    def forward(self, x):
        fft_x = self.conv_fft(x)
        b,c,h,w = fft_x.shape
        fft_map = torch.zeros(b, 32, h, w)
        for i in range(b):
            
            img = fft_x[i, :, :, :].cpu().detach().numpy().transpose(1,2,0) # b c h w -> h w c
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fft = fft_img(img*255)/255
            fft = torch.cat([fft for i in range(32)], dim=0)
            
            fft_map[i, :,:,:] = fft
        x = self.conv_large(x)
        x = self.block_t(x)
        x = self.generator(x, fft_map)   
        x = self.gm(x)
        x = self.conv_A(x)
        return x  



