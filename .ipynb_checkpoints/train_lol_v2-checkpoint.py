import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
from torch.nn.parallel import DataParallel

import datetime
import os
import sys
import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.models import vgg16

from data_loaders.lol import lowlight_loader
from model.IAT_main import IAT
from model.Baseline import Baseline
from model.Net1 import Net1
from model.Net2 import Net2
from model.Net3 import Net3
from model.Net4 import Net4
from model.Net5 import Net5
from model.Net import Net


import smtplib
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText
from email.header import Header



from IQA_pytorch import SSIM
from utils import PSNR, adjust_learning_rate, validation, LossNetwork, visualization

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default='0, 1')
parser.add_argument('--img_path', type=str, default="/data/unagi0/cui_data/light_dataset/LOL/Train/Low/")
parser.add_argument('--img_val_path', type=str, default="/data/unagi0/cui_data/light_dataset/LOL/Test/Low/")
parser.add_argument("--normalize", action="store_false", help="Default Normalize in LOL training.")
parser.add_argument('--model_type', type=str, default='s')
parser.add_argument('--model', type=str, default='IAT')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--pretrain_dir', type=str, default=None)

parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--display_iter', type=int, default=10)
parser.add_argument('--snapshots_folder', type=str, default="workdirs/snapshots_folder_lol")

config = parser.parse_args()

print(config)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

if not os.path.exists(config.snapshots_folder):
    os.makedirs(config.snapshots_folder)

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
elif config.model == 'Net4':
    model = Net4().cuda()
elif config.model == 'Net5':
    model = Net5().cuda()
elif config.model == 'Net':
    model = Net().cuda()  

model = DataParallel(model, device_ids=[0, 1])

if config.pretrain_dir is not None:
    model.load_state_dict(torch.load(config.pretrain_dir))

# Data Setting

train_dataset = lowlight_loader(images_path=config.img_path, normalize=config.normalize)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8,
                                           pin_memory=True)

val_dataset = lowlight_loader(images_path=config.img_val_path, mode='test', normalize=config.normalize)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

# Loss & Optimizer Setting & Metric
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.cuda()

for param in vgg_model.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

# optimizer = torch.optim.Adam(model.parameters(),
#                 lr=config.lr,
#                 betas=(0.9, 0.999),
#                 eps=1e-08,
#                 amsgrad=False)


# optimizer = torch.optim.Adam([{'params': model.global_net.parameters(),'lr':config.lr*0.1},
#             {'params': model.local_net.parameters(),'lr':config.lr}], lr=config.lr, weight_decay=config.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

device = next(model.parameters()).device
print('the device is:', device)

L1_loss = nn.L1Loss()
L1_smooth_loss = F.smooth_l1_loss

loss_network = LossNetwork(vgg_model)
loss_network.eval()

ssim = SSIM()
psnr = PSNR()
ssim_high = 0
psnr_high = 0
now_time = datetime.datetime.now()
month = now_time.month
day = now_time.day
minute = now_time.minute

model.train()
print('######## Start IAT Training #########')
for epoch in range(config.num_epochs):
    # adjust_learning_rate(optimizer, epoch)
    print('the epoch is:', epoch)
    for iteration, imgs in enumerate(train_loader):
        low_img, high_img = imgs[0].cuda(), imgs[1].cuda()
        # Checking!
        #visualization(low_img, 'show/low', iteration)
        #visualization(high_img, 'show/high', iteration)
        optimizer.zero_grad()
        model.train()
        mul, add, enhance_img = model(low_img)

        loss = L1_smooth_loss(enhance_img, high_img)+0.04*loss_network(enhance_img, high_img)
        
        loss.backward()
        
        optimizer.step()
        scheduler.step()

        if ((iteration + 1) % config.display_iter) == 0:
            print("Loss at iteration", iteration + 1, ":", loss.item())

    # Evaluation Model
    model.eval()
    SSIM_mean, PSNR_mean  = validation(model, val_loader)

    with open(config.snapshots_folder + '/log.txt', 'a+') as f:
        f.write('epoch' + str(epoch) + ':' + 'the SSIM is' + str(SSIM_mean) + 'the PSNR is' + str(PSNR_mean) + '\n')

    if SSIM_mean > ssim_high:
        ssim_high = SSIM_mean
        print('the highest SSIM value is:', str(ssim_high))
        torch.save(model.state_dict(), os.path.join(config.snapshots_folder, f"best_Epoch_ssim" + '.pth'))
        
    if PSNR_mean > psnr_high:
        psnr_high = PSNR_mean
        print('the highest PSNR value is:', str(PSNR_mean))
        torch.save(model.state_dict(), os.path.join(config.snapshots_folder, f"best_Epoch_psnr" + '.pth'))

    f.close()

