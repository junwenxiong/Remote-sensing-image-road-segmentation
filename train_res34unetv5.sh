CUDA_VISIBLE_DEVICES=1 python train_res18.py --backbone res34unetv5 --dyrelu True --dataset ./huawei/new_data/512 --workers 8 \
--epochs 100 --batch-size 3 --learn-rate 1e-4 --loss dice_bce  --optim adam --iter-num 1000