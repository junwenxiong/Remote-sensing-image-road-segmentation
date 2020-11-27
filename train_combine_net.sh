CUDA_VISIBLE_DEVICES=0 python train_combine_net.py --backbone combinenet --dataset ./huawei --workers 8 \
--epochs 100 --batch-size 4 --learn-rate 1e-4 --optim ranger  --combine True  --train False \
--model1 resunet34 --model1-checkpoint ./weights/resunet34/ResNet34Unet55.pth \
--model2 res34unetv5 --model2-checkpoint ./weights/res34unetv5/res34unetv535.pth