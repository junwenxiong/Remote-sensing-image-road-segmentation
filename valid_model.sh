CUDA_VISIBLE_DEVICES=0 python valid_model.py --backbone resunet34 --dataset ./huawei --workers 8 \
--epochs 100 --batch-size 8 --learn-rate 1e-4 --optim ranger --test True