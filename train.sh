CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --backbone resnet --lr 0.01 --workers 4 --epochs 40 --batch-size 16 --gpu-ids 0,1,2,3 --checkname unet-resnet --eval-interval 1
