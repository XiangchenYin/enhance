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
from model.Net4 import Net4
from model.Net5 import Net5
from model.Net import Net


from IQA_pytorch import SSIM, MS_SSIM
from data_loaders.lol import lowlight_loader
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=0)
parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--img_val_path', type=str, default="/data/unagi0/cui_data/light_dataset/LOL_v2/Test/Low/")
parser.add_argument('--checkpoint', type=str, default="workdirs/snapshots_folder_lol/best_Epoch.pth")
parser.add_argument('--model', type=str, default='IAT')
parser.add_argument('--pre_norm', type=bool, default=True)
config = parser.parse_args()

print(config)
val_dataset = lowlight_loader(images_path=config.img_val_path, mode='test', normalize=config.pre_norm)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

# Model Setting
if config.model == 'IAT':
    model = IAT().cuda()
elif config.model == 'Baseline':
    model = Baseline().cuda()
elif config.model == 'Net1':
    model = Net1().cuda()
elif config.model == 'Net2':
    model = Net2().cuda()
elif config.model == 'Net3':
    model = Net3().cuda()
elif config.model == 'Net4':
    model = Net4().cuda()
elif config.model == 'Net5':
    model = Net5().cuda()
elif config.model == 'Net':
    model = Net().cuda()    

model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(config.checkpoint).items()})



# model.load_state_dict(torch.load(config.checkpoint))
model.eval()


ssim = SSIM()
psnr = PSNR()
ssim_list = []
psnr_list = []

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if config.save:
    result_path = config.img_val_path.replace('Low', 'Result')
    mkdir(result_path)

with torch.no_grad():
    for i, imgs in tqdm(enumerate(val_loader)):
        #print(i)
        low_img, high_img = imgs[0].cuda(), imgs[1].cuda()
        #print(low_img.shape)
        mul, add ,enhanced_img = model(low_img)

        torchvision.utils.save_image(enhanced_img, result_path + str(i) + '.png')

        ssim_value = ssim(enhanced_img, high_img, as_loss=False).item()
        psnr_value = psnr(enhanced_img, high_img).item()

        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)


SSIM_mean = np.mean(ssim_list)
PSNR_mean = np.mean(psnr_list)
print('The SSIM Value is:', SSIM_mean)
print('The PSNR Value is:', PSNR_mean)
