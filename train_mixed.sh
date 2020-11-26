CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_mixed.py \
--backbone resunet34 --dyrelu False --dataset ./huawei/new_data/512 --workers 8 \
--epochs 100 --batch-size 2 --learn-rate 1e-4 --loss dice_bce  --optim adam --iter-num 1000 \
--mixed-train True --train False --gpu-ids 0,1