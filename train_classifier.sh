CUDA_VISIBLE_DEVICES=0 python train_classifier.py  --dataset ./huawei --workers 8 \
--num-epochs 100 --batch-size 64 --lr 1e-4 
