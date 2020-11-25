CUDA_VISIBLE_DEVICES=0 python train_combine_net.py --backbone combinenet --dataset ./huawei --workers 8 \
--epochs 100 --batch-size 4 --learn-rate 1e-4 --optim ranger  --combine True  \
--model1 resxtunet34 --model1-checkpoint ./weights/resxtunet34/resxtunet3440.pth \
--model2 res34unetv5 --model2-checkpoint ./weights/res34unetv5/model-best.pth