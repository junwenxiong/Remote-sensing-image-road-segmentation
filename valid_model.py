import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
from tqdm import tqdm
import cv2
import os
import numpy as np
from time import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101
from networks.linknet34 import LinkNet34
from networks.baselineZ import UNet
from utils.framework import MyFrame
from utils.loss import dice_bce_loss
from utils.data import ImageFolder
from networks.Resnet18Unet import ResNet18Unet, ResNet34Unet, ResNeXt50Unet
from networks.unet_model import Res34Unetv3, Res34Unetv4, Res34Unetv5
from networks.unet_att_Dyrelu import UNet_att_Dyre
from networks.dlinknet import DLinkNet50
from networks.CombineNet import CombineNet
from networks.unet import Unet
from utils.args_config import make_args
from utils.loss_2 import SegmentationLosses
from random import shuffle

args = make_args()
print(args)

# val_ROOT = '/data/home/xjw/dataset/Huawei/data/val/'
# val_image_dir = '/data/home/xjw/dataset/Huawei/data/val/images/'

val_ROOT = os.path.join(args.dataset, 'val')
val_image_dir = os.path.join(val_ROOT, 'images')

# import pdb 
# pdb.set_trace()
val_list = os.listdir(val_image_dir)
shuffle(val_list)
x, y, c = cv2.imread(os.path.join(val_image_dir, val_list[0])).shape
SHAPE = (x, y)

# NAME = args.backbone

NAME = 'ResNet34UnetDyrelu'

BATCHSIZE_PER_CARD = 1

batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

val_dataset = ImageFolder(val_list, val_ROOT)
val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.workers)


mylog = open('logs/' + NAME + '_Test.log', 'w')
tic = time()

model_dir = './weights/' + NAME + '/'
model_list = os.listdir(model_dir)
sort_num_list = []
for file in model_list:
    sort_num_list.append(int(file.split(NAME)[1].split('.pth')[0]))  # 去掉前面的字符串和下划线以及后缀，只留下数字并转换为整数方便后面排序
    # sort_num_list.append(file)
sort_num_list.sort()  # 然后再重新排序

# print(sort_num_list)
# 接着再重新排序读取文件
# import pdb
# pdb.set_trace()
sorted_model_file = []
for sort_num in sort_num_list:
    for file in model_list:
        if sort_num == int(file.split(NAME)[1].split('.pth')[0]):
            sorted_model_file.append(file)

# import pdb
# pdb.set_trace()
for model_path in sorted_model_file:
   
    solver = MyFrame(args, )
    solver.load(str(model_dir + model_path))

    val_data_loader_iter = iter(val_data_loader)
    val_epoch_loss = 0
    val_miou = 0.0
    val_acc = 0.0
    print("validing:")
    for img, mask in tqdm(val_data_loader_iter):
        solver.set_input(img, mask)
        miou, acc, val_loss = solver.val_optimize()
        val_miou += miou
        val_acc += acc
        val_epoch_loss += val_loss

    val_epoch_loss /= len(val_data_loader_iter)
    val_miou /= len(val_data_loader_iter)
    val_acc /= len(val_data_loader_iter)

    print('model_name:',
          model_path,
          '    time:',
          int(time() - tic),
          file=mylog)
    print('val_miou',
          val_miou,
          'val_acc',
          val_acc,
          'val_loss:',
          val_epoch_loss,
          file=mylog)
    print('********', file=mylog)
    print('model_name:', model_path)
    print('model_name:', model_path, '    time:', int(time() - tic))
    print('val_miou', val_miou, 'val_acc', val_acc, 'val_loss:',
          val_epoch_loss)

    mylog.flush()

print('Finish!', file=mylog)
print('Finish!')
mylog.close()