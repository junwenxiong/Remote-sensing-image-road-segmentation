CUDA_VISIBLE_DEVICES=1 python train_res18.py --backbone resxtunet34 --dataset ./huawei --workers 8 \
--epochs 100 --batch-size 12 --learn-rate 1e-4 --optim ranger