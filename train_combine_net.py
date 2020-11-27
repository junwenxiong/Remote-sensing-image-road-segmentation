import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import cv2
import os
import numpy as np
from PIL import Image
from time import time, strftime, localtime
from tensorboardX import SummaryWriter
from torch.autograd import Variable as V
from utils.framework import MyFrame
from utils.data import ImageFolder, make_dataloader
from utils.args_config import make_args

args = make_args()
print(args)

#训练测试 256*256
NAME = args.backbone
now = strftime('%m-%d-%H-%M', localtime())
model_dir = './weights/' + NAME + now + '/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

solver = MyFrame(args, )

train_ROOT = os.path.join(args.dataset, 'train_new')
val_ROOT = os.path.join(args.dataset, 'val_new')
test_ROOT = os.path.join(args.dataset, 'test_new')

train_image_dir = os.path.join(train_ROOT, 'images')
train_label_dir = os.path.join(train_ROOT, 'labels')

val_image_dir = os.path.join(val_ROOT, 'images')
val_label_dir = os.path.join(val_ROOT, 'labels')

train_list = os.listdir(train_image_dir)
val_list = os.listdir(val_image_dir)
test_image_dir = os.path.join(test_ROOT, 'images')
test_list = os.listdir(test_image_dir)


mylog = open('logs/' + NAME + '.log', 'w')
print(args, file=mylog)
tic = time()
no_optim = 0
total_epoch = args.epochs
train_epoch_best_loss = 100.

writer = SummaryWriter()

it_train_num = 0


def valid(epoch):
    # *****验证*******
    num_img_tr = len(val_list)
    tbar = tqdm(val_list, desc='\r')
    val_epoch_loss = 0
    val_miou = 0.0
    val_acc = 0.0
    it_val_num = 0
    print("validing:")
    for i, filename in tbar:
        it_val_num += 1
        image = Image.open(os.path.join(val_image_dir, filename))
        label = Image.open(os.path.join(val_label_dir, filename))
        label = np.array(label).astype(np.float32)
        image = np.array(image, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
        label = np.array(label, np.float32).transpose(2, 0, 1) / 255.0
        label = torch.from_numpy(label).float()
        label = label.cuda()

        solver.set_combine_input(image, label)
        miou, acc, val_loss = solver.ensemble_val_optimize()
        
        val_miou += miou
        val_acc += acc
        val_epoch_loss += val_loss
        
    val_epoch_loss /= num_img_tr
    val_miou /= num_img_tr
    val_acc /= num_img_tr
    writer.add_scalar('dataset/val_epoch_loss', val_epoch_loss, epoch)
    writer.add_scalar('dataset/val_miou', val_miou, epoch)

    print('val_epoch:',
          epoch,
          file=mylog)
    print('val_miou',
          val_miou,
          'val_acc',
          val_acc,
          'val_loss:',
          val_epoch_loss,
          file=mylog)
    print('********', file=mylog)

    print('val_epoch:', epoch,)
    print('val_miou', val_miou, 'val_acc', val_acc, 'val_loss:',
          val_epoch_loss)


for epoch in range(0, total_epoch + 1):
    train_epoch_loss = 0
    num_img_tr = len(train_list)
    tbar = tqdm(train_list, desc='\r')
    it_train_num = 0
    for i, filename in enumerate(tbar):
        it_train_num += 1
        image = Image.open(os.path.join(train_image_dir, filename))
        label = Image.open(os.path.join(train_label_dir, filename))
        label = np.array(label).astype(np.float32)
        image = np.array(image, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
        label = np.array(label, np.float32).transpose(2, 0, 1) / 255.0
        label = torch.from_numpy(label).float()
        label = label.cuda()

        solver.set_combine_input(image, label)
        train_loss = solver.ensemble_optimize()
        train_epoch_loss += train_loss

    valid(epoch)

    train_epoch_loss /= num_img_tr
    writer.add_scalar('dataset/train_epoch_loss', train_epoch_loss, epoch)
    print('*********', file=mylog)
    print('epoch:', epoch,  file=mylog)
    print('train_loss:', train_epoch_loss, file=mylog)

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.combine_save(model_dir + NAME + '%d.pth' % epoch)
        str = NAME + '%d.pth' % epoch
    if no_optim > 8:
        print('early stop at %d epoch', file=mylog)

        print('early stop at %d epoch' % epoch)
        break
    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        solver.combine_load(model_dir + str)
    solver.update_lr(5.0, factor=True, mylog=mylog)

    mylog.flush()

print('Finish!', file=mylog)
print('Finish!')
mylog.close()