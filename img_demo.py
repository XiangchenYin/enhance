import os 
import torch
import cv2
import argparse
import warnings
import torchvision
import numpy as np
from utils import PSNR, validation, LossNetwork
from model.IAT_main import IAT
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt
from PIL import Image
from model.Net5 import Net5

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, default='/root/autodl-tmp/Exdark/JPEGImages/IMGS')
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--task', type=str, default='enhance', help='Choose from exposure or enhance')
config = parser.parse_args()

# Weights path
exposure_pretrain = r'best_Epoch_exposure.pth'
enhance_pretrain = r'workdirs/Net5_lolv2-2gpus/best_Epoch-6531.pth'

normalize_process = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

## Load Pre-train Weights
model = Net5().cuda()


if config.task == 'exposure':
    model.load_state_dict(torch.load(exposure_pretrain))
elif config.task == 'enhance':
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(enhance_pretrain).items()})
#     model.load_state_dict(torch.load(enhance_pretrain))
else:
    warnings.warn('Only could be exposure or enhance')
model.eval()

save_dir = './IMGS_Net5'

if os.path.isfile(config.file_name):
    ## Load Image
    img = Image.open(config.file_name)
#     img=img.resize((608,608), Image.ANTIALIAS)
    img = (np.asarray(img)/ 255.0)
    if img.shape[2] == 4:
        img = img[:,:,:3]
    input = torch.from_numpy(img).float().cuda()
    input = input.permute(2,0,1).unsqueeze(0)
    if config.normalize:    # False
        input = normalize_process(input)

    ## Forward Network
    _, _ ,enhanced_img = model(input)

    torchvision.utils.save_image(enhanced_img, 'result.png')
else:
    
    image_list = os.listdir(config.file_name)
    for i in image_list:
        image_name = os.path.join(config.file_name, i)
    ## Load Image
        img = Image.open(image_name)
        w, h  = img.size
        img=img.resize((608,608), Image.ANTIALIAS)
            
        img = (np.asarray(img)/ 255.0)
        
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
            img = np.concatenate((img,img,img), axis=2)
            
        if img.shape[2] == 4:
            img = img[:,:,:3]
            
        input = torch.from_numpy(img).float().cuda()
        input = input.permute(2,0,1).unsqueeze(0)
        
        if config.normalize:    # False
            input = normalize_process(input)
        
        ## Forward Network
        _, _ , enhanced_img = model(input)
        
#         enhanced_img = enhanced_img.squeeze(0).cpu()
#         enhanced_img = torchvision.transforms.ToPILImage()(enhanced_img)
#         enhanced_img = enhanced_img.resize((w,h), Image.ANTIALIAS)
#         enhanced_img.save(os.path.join(save_dir, i))
        enhanced_img = enhanced_img.squeeze(0).cpu().detach().numpy().transpose(1,2,0)
        enhanced_img = enhanced_img *255
        enhanced_img = cv2.resize(enhanced_img, (w, h), cv2.INTER_LINEAR)        
        cv2.imwrite(os.path.join(save_dir, i), enhanced_img)
        
#         torchvision.utils.save_image(enhanced_img, os.path.join(save_dir, i))
        print(f'{i} is enhanced!')