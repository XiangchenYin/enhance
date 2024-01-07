import cv2
import pyiqa
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F

import os
import argparse
import numpy as np
from utils import PSNR, validation, LossNetwork
from model.IAT_main import IAT
from model.Baseline import Baseline
from model.Net1 import Net1
from model.Net2 import Net2
from model.Net3 import Net3
from model.Net import Net
import lpips


from IQA_pytorch import SSIM, MS_SSIM, LPIPSvgg
# from data_loaders.lol_v1_new import lowlight_loader_new
from pyiqa.archs import niqe_arch
from data_loaders.lol_v1_whole import lowlight_loader_new


from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=0)
parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--model', type=str, default='IAT')


parser.add_argument('--img_val_path', type=str, default='/root/autodl-tmp/LOLv1/eval15/low/')
parser.add_argument('--checkpoint_path', type=str, default='workdirs/Baseline_LOLv1_first/best_Epoch_psnr.pth')
config = parser.parse_args()

print(config)
val_dataset = lowlight_loader_new(images_path=config.img_val_path, mode='test')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)


# Model Setting
if config.model == 'IAT':
    model = IAT(type='ccc').cuda()
elif config.model == 'Baseline':
    model = Baseline().cuda()
elif config.model == 'Net1':
    model = Net1().cuda()
elif config.model == 'Net2':
    model = Net2().cuda()
elif config.model == 'Net3':
    model = Net3().cuda()
elif config.model == 'Net':
    model = Net().cuda()


model.load_state_dict(torch.load(config.checkpoint_path))
model.eval()


lpips_vgg = LPIPSvgg().cuda()
ssim = SSIM()
psnr = PSNR()
ssim_list = []
psnr_list = []
lpips_list = []
niqe_list = []

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if config.save:
    result_path = config.img_val_path.replace('low', 'Result')
    mkdir(result_path)

with torch.no_grad():
    for i, imgs in tqdm(enumerate(val_loader)):
        #print(i)
        low_img, high_img, name = imgs[0].cuda(), imgs[1].cuda(), str(imgs[2][0])
        print(name)
        #print(low_img.shape)
        mul, add ,enhanced_img = model(low_img)
        if config.save:
            torchvision.utils.save_image(enhanced_img, result_path + str(name) + '.png')

        ssim_value = ssim(enhanced_img, high_img, as_loss=False).item()
        psnr_value = psnr(enhanced_img, high_img).item()
        lpips_value = lpips_vgg(enhanced_img, high_img).item()
        # enhanced_img = enhanced_img.unqueeze(0).cpu().numpy().permute(1, 2, 0)
        import pyiqa
        iqa = pyiqa.create_metric('niqe').cuda()

        niqe_value = iqa(enhanced_img).item()

        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)
        lpips_list.append(lpips_value)
        niqe_list.append(niqe_value)



SSIM_mean = np.mean(ssim_list)
PSNR_mean = np.mean(psnr_list)
LPIPS_mean = np.mean(lpips_list)
NIQE_mean = np.mean(niqe_list)

print('The SSIM Value is:', SSIM_mean)
print('The PSNR Value is:', PSNR_mean)
print('The LPIPS Value is:', LPIPS_mean)
print('The NIQE Value is:', NIQE_mean)


