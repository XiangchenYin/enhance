import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models.vgg as vgg
import torchvision.transforms as transforms
import cv2
import pdb
import math
import numpy as np
#from  siamese import SiameseNet
import pickle
import os
import matplotlib 
matplotlib.rcParams['backend'] = "Agg" 


# 代码来源： https://github.com/hhb072/WaveletSRNet
# Donwload the wavelet parameters from [ https://github.com/hhb072/WaveletSRNet/blob/master/wavelet_weights_c2.pkl ] wavelet_weights_c2.pkl.


class WaveletTransform(nn.Module): 
    def __init__(self, scale=1, dec=True, params_path='model/wavelet_weights_c2.pkl', transpose=True):
        super(WaveletTransform, self).__init__()
        
        self.scale = scale
        self.dec = dec
        self.transpose = transpose
        
        ks = int(math.pow(2, self.scale)  )
        nc = 3 * ks * ks
        
        if dec:
          self.conv = nn.Conv2d(in_channels=16, out_channels=nc, kernel_size=ks, stride=ks, padding=0, groups=16, bias=False)
        else:
          self.conv = nn.ConvTranspose2d(in_channels=nc, out_channels=16, kernel_size=ks, stride=ks, padding=0, groups=16, bias=False)


#           self.conv = nn.ConvTranspose2d(in_channels=nc, out_channels=3, kernel_size=ks, stride=ks, padding=0, groups=3, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                f = open(params_path,'rb')
                dct = pickle.load(f, encoding='latin1')
                f.close()
                m.weight.data = torch.from_numpy(dct['rec%d' % ks])
                m.weight.requires_grad = False  
                           
    def forward(self, x): 
        if self.dec:
          output = self.conv(x)          
          if self.transpose:
            osz = output.size()
            #print(osz)
            output = output.view(osz[0], 3, -1, osz[2], osz[3]).transpose(1,2).contiguous().view(osz)            
        else:
          if self.transpose:
            xx = x
            xsz = xx.size()
            xx = xx.view(xsz[0], -1, 3, xsz[2], xsz[3]).transpose(1,2).contiguous().view(xsz)             
          output = self.conv(xx)        
        return output 

    
    
if __name__ == '__main__':
     
    wavelet_dec = WaveletTransform(scale=2, dec=True)    ## 小波变换
    wavelet_rec = WaveletTransform(scale=2, dec=False)   ## 小波逆变换
#     img = cv2.imread('22.png')
    
#     img = transforms.ToTensor()(img).unsqueeze(0)

    img = torch.ones(1,16,128,128)
    
    print(img.shape)
    w_img = wavelet_dec(img)
    print(w_img.shape)
    r_img = wavelet_rec(w_img)
    print(r_img.shape)
    result = r_img.squeeze(0).numpy().transpose(1,2,0)
    print(result.shape)
    cv2.imwrite('result.png', result*255)
    
    
    




        


        