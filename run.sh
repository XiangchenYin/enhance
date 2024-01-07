python train_lol_v1_whole.py --gpu_id 0 --snapshots_folder workdirs/Wave_MIT --model Net1 --batch_size 4 --num_epochs 1000 \
--img_path /home/yin/Workspace/data/MIT-Adobe-5K-512/train/low --img_val_path /home/yin/Workspace/data/MIT-Adobe-5K-512/test/low \
--model Net3


python evaluation_lol_v1.py   --model Net3 --checkpoint_path workdirs/Wave_MIT/best_Epoch_ssim.pth --img_val_path /home/yin/Workspace/data/MIT-Adobe-5K-512/test/low


####
python evaluation_lol_v1.py   --model Net3 --checkpoint_path workdirs/Wave_MIT/best_Epoch_ssim.pth --img_val_path /home/yin/Workspace/data/LOLv1/eval15/low

python evaluation_lol_v1.py   --model Net3 --checkpoint_path workdirs/Net_LOLv1_400x600/best_Epoch_ssim.pth --img_val_path  /home/yin/Workspace/data/MIT-Adobe-5K-512/test/low
####
python evaluation_lol_v1.py   --model Net --checkpoint_path workdirs/Net_LOLv1_400x600/best_Epoch_ssim.pth --img_val_path /home/yin/Workspace/data/LOLv1/eval15/low

python evaluation_lol_v1.py   --model Net --checkpoint_path workdirs/Net_lolv2_final/best_Epoch_psnr.pth --img_val_path /home/yin/Workspace/data/LOLv1/eval15/low

python evaluation_lol_v2.py --img_val_path /home/yin/Workspace/data/LOL_v2/Test/Low/ --checkpoint workdirs/Net_lolv2_final/best_Epoch_ssim.pth --model Net


python train_lol_v2.py --gpu_id 0,1 --img_path /root/autodl-tmp/LOL_v2/Train/Low/ --img_val_path /root/autodl-tmp/LOL_v2/Test/Low/ --snapshots_folder workdirs/Net_lolv2_final --model Net


python train_lol_v1_whole.py --gpu_id 0 --snapshots_folder workdirs/Baseline_LOLv1_600x600 --model Baseline --batch_size 8 

python train_lol_v1_whole.py --gpu_id 0 --snapshots_folder workdirs/Net1_LOLv1_400x600 --model Net1 --batch_size 8 --num_epochs 1000

python train_lol_v1_whole.py --gpu_id 0 --snapshots_folder workdirs/Net2_LOLv1_400x600 --model Net2 --batch_size 4 --num_epochs 1000

python train_lol_v1_whole.py --gpu_id 0 --snapshots_folder workdirs/Net3_LOLv1_400x600 --model Net3 --batch_size 4 --num_epochs 1000

python evaluation_lol_v1.py   --model Baseline --checkpoint_path workdirs/Baseline_LOLv1_400x600/best_Epoch_psnr.pth

python evaluation_lol_v1.py   --model Net1 --checkpoint_path workdirs/Net1_LOLv1_400x600/best_Epoch_psnr.pth

python evaluation_lol_v1.py   --model Net2 --checkpoint_path workdirs/Net2_LOLv1_400x600/best_Epoch_psnr.pth

python evaluation_lol_v1.py   --model Net3 --checkpoint_path workdirs/Net3_LOLv1_400x600/best_Epoch_psnr.pth

python evaluation_lol_v1.py   --model Net --checkpoint_path workdirs/Net_LOLv1_400x600/Epoch_943.pth


python evaluation_lol_v2.py --img_val_path /home/yin/Desktop/data/LOL_v2/Test/Low/

python LOL_patch.py --src_dir /home/yin/Desktop/data/LOL_v1/our485 --tar_dir /home/yin/Desktop/data/LOL_v1/our485_patch

python train_lol_v1_patch.py --img_path /home/yin/Desktop/data/LOL_v1/our485_patch/low/ --img_val_path /home/yin/Desktop/data/LOL_v1/eval15/low/
python train_lol_v1_whole.py --img_path /home/yin/Desktop/data/LOL_v1/our485/low/ --img_val_path /home/yin/Desktop/data/LOL_v1/eval15/low/ --pretrain_dir workdirs/snapshots_folder_lol_v1_patch/best_Epoch.pth

python evaluation_lol_v1.py --img_val_path /home/yin/Desktop/data/LOL_v1/eval15/low/

python train_lol_v2.py --gpu_id 0 --img_path /root/autodl-tmp/LOL_v2/Train/Low/ --img_val_path /root/autodl-tmp/LOL_v2/Test/Low/

python evaluation_lol_v2.py --img_val_path /root/autodl-tmp/LOL_v2/Test/Low/ --checkpoint workdirs/snapshots_folder_lol/best_Epoch.pth

python evaluation_lol_v2.py --img_val_path /root/autodl-tmp/LOL_v2/Test/Low/ 

python train_lol_v2.py --gpu_id 0 --img_path /root/autodl-tmp/LOL_v2/Train/Low/ --img_val_path /root/autodl-tmp/LOL_v2/Test/Low/ --snapshots_folder workdirs/snapshots_folder_lol

python train_lol_v2.py --gpu_id 1 --img_path /root/autodl-tmp/LOL_v2-copy/Train/Low/ --img_val_path /root/autodl-tmp/LOL_v2-copy/Test/Low/ --snapshots_folder workdirs/IAT_lolv2 --model IAT



python train_lol_v2.py --gpu_id 0 --img_path /root/autodl-tmp/LOL_v2/Train/Low/ --img_val_path /root/autodl-tmp/LOL_v2/Test/Low/ --snapshots_folder workdirs/Baseline_lolv2 --model Baseline

python evaluation_lol_v2.py --img_val_path /root/autodl-tmp/LOL_v2/Test/Low/ --checkpoint workdirs/Baseline_lolv2/best_Epoch-52449.pth --model Baseline

python evaluation_lol_v2.py --img_val_path /root/autodl-tmp/LOL_v2/Test/Low/ --checkpoint workdirs/Baseline_lolv2/best_Epoch-52553.pth --model Baseline


python train_lol_v2.py --gpu_id 0 --img_path /root/autodl-tmp/LOL_v2/Train/Low/ --img_val_path /root/autodl-tmp/LOL_v2/Test/Low/ --snapshots_folder workdirs/Net1_lolv2 --model Net1

python train_lol_v2.py --gpu_id 0 --img_path /root/autodl-tmp/LOL_v2-copy/Train/Low/ --img_val_path /root/autodl-tmp/LOL_v2-copy/Test/Low/ --snapshots_folder workdirs/Net2_lolv2 --model Net2

python train_lol_v2.py --gpu_id 1 --img_path /root/autodl-tmp/LOL_v2/Train/Low/ --img_val_path /root/autodl-tmp/LOL_v2/Test/Low/ --snapshots_folder workdirs/Net3_lolv2 --model Net3

python train_lol_v2.py --gpu_id 0 --img_path /root/autodl-tmp/LOL_v2/Train/Low/ --img_val_path /root/autodl-tmp/LOL_v2/Test/Low/ --snapshots_folder workdirs/Net4_lolv2 --model Net4 --batch_size 4 --lr 0.0001

python train_lol_v2.py --gpu_id 0 --img_path /root/autodl-tmp/LOL_v2/Train/Low/ --img_val_path /root/autodl-tmp/LOL_v2/Test/Low/ --snapshots_folder workdirs/Net5_lolv2 --model Net5 --batch_size 4

python train_lol_v2.py --gpu_id 0,1 --img_path /root/autodl-tmp/LOL_v2/Train/Low/ --img_val_path /root/autodl-tmp/LOL_v2/Test/Low/ --snapshots_folder workdirs/Net4_lolv2-2gpus --model Net4 --batch_size 8 


python train_lol_v2.py --gpu_id 0,1 --img_path /root/autodl-tmp/LOL_v2/Train/Low/ --img_val_path /root/autodl-tmp/LOL_v2/Test/Low/ --snapshots_folder workdirs/Net5_lolv2-2gpus --model Net5 --batch_size 8


# error
python train_lol_v2.py --gpu_id 0,1 --img_path /root/autodl-tmp/LOL_v2/Train/Low/ --img_val_path /root/autodl-tmp/LOL_v2/Test/Low/ --snapshots_folder workdirs/Net5_lolv2-new --model Net5 --batch_size 8

python train_lol_v2.py --gpu_id 0,1 --img_path /root/autodl-tmp/LOL_v2/Train/Low/ --img_val_path /root/autodl-tmp/LOL_v2/Test/Low/ --snapshots_folder workdirs/Net6_lolv2-2gpus --model Net6 --batch_size 8


python evaluation_lol_v2.py --img_val_path /root/autodl-tmp/LOL_v2/Test/Low/ --checkpoint workdirs/Net1_lolv2/best_Epoch-52925.pth --model Net1

python evaluation_lol_v2.py --img_val_path /root/autodl-tmp/LOL_v2/Test/Low/ --checkpoint workdirs/Net2_lolv2/best_Epoch-52927.pth --model Net2

python evaluation_lol_v2.py --img_val_path /root/autodl-tmp/LOL_v2/Test/Low/ --checkpoint workdirs/Net3_lolv2/best_Epoch-52938.pth --model Net3

python evaluation_lol_v2.py --img_val_path /root/autodl-tmp/LOL_v2/Test/Low/ --checkpoint workdirs/Net4_lolv2-2gpus/best_Epoch-6446.pth --model Net4

python evaluation_lol_v2.py --img_val_path /root/autodl-tmp/LOL_v2/Test/Low/ --checkpoint workdirs/Net5_lolv2-2gpus/best_Epoch-6531.pth --model Net5 


python evaluation_lol_v2.py --img_val_path /root/autodl-tmp/LOL_v2/Test/Low/ --checkpoint best_Epoch_lol.pth --model IAT


python demo/image_demo.py 图片3.png work_dirs/Net5-joint/Net5.py work_dirs/Net5-joint/epoch_40.pth





