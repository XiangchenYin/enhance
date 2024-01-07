import torch.nn as nn
import torch
from torchvision import transforms
import cv2
import numpy as np

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
        self.generator = CA(in_dim=out_channels)
        
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_fft = nn.Conv2d(3, 3, kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.conv_A = nn.Conv2d(out_channels, 3, kernel_size=3, stride=(1, 1), padding=(1, 1))



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
        x = self.generator(x, fft_map)      
        x = self.conv_A(x)
        return x  