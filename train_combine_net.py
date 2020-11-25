import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import cv2
import os
import numpy as np
from time import time, strftime, localtime
from tensorboardX import SummaryWriter
from torch.autograd import Variable as V
from utils.framework import MyFrame
from utils.data import ImageFolder
from utils.args_config import make_args

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = make_args()
print(args)

#训练测试 256*256
train_ROOT = os.path.join(args.dataset, 'train')
val_ROOT = os.path.join(args.dataset, 'val')

train_image_dir = os.path.join(train_ROOT, 'images')
val_image_dir = os.path.join(val_ROOT, 'images')

train_list = os.listdir(train_image_dir)
val_list = os.listdir(val_image_dir)
x, y, c = cv2.imread(os.path.join(train_image_dir, train_list[0])).shape
SHAPE = (x, y)

NAME = args.backbone
model_dir = './weights/' + NAME + '/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

BATCHSIZE_PER_CARD = args.batch_size

solver = MyFrame(
    args,
)

batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

dataset = ImageFolder(train_list[:50000], train_ROOT)
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batchsize,
                                          shuffle=True,
                                          num_workers=args.workers)

val_dataset = ImageFolder(val_list, val_ROOT)
val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=batchsize,
                                              shuffle=True,
                                              num_workers=args.workers)

now = strftime('%m-%d-%H-%M', localtime())
mylog = open('logs/' + NAME + now + '.log', 'w')
print(args, file=mylog)
tic = time()
no_optim = 0
total_epoch = args.epochs
train_epoch_best_loss = 100.
str = ""
writer = SummaryWriter()

it_train_num = 0


def valid(epoch, it_train_num):
    # *****验证*******
    val_data_loader_iter = iter(val_data_loader)
    val_epoch_loss = 0
    val_miou = 0.0
    val_acc = 0.0
    it_val_num = 0
    print("validing:")
    for img, mask in tqdm(val_data_loader_iter):
        it_val_num += 1
        solver.set_input(img, mask)
        miou, acc, val_loss = solver.combine_val_optimize()
        val_miou += miou
        val_acc += acc
        val_epoch_loss += val_loss
        writer.add_scalar('dataset/val_loss%d' % it_train_num, val_loss,
                          it_val_num)
        writer.add_scalar('dataset/val_loss%d' % epoch, val_loss, it_val_num)
        writer.add_scalar('dataset/miou%d' % it_train_num, miou, it_val_num)

    val_epoch_loss /= len(val_data_loader_iter)
    val_miou /= len(val_data_loader_iter)
    val_acc /= len(val_data_loader_iter)
    writer.add_scalar('dataset/val_epoch_loss', val_epoch_loss, epoch)
    writer.add_scalar('dataset/val_miou', val_miou, epoch)

    print('val_epoch:',
          epoch,
          'it_train_num:',
          it_train_num,
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

    print('val_epoch:', epoch, 'it_train_num:', it_train_num, '    time:',
          int(time() - tic))
    print('val_miou', val_miou, 'val_acc', val_acc, 'val_loss:',
          val_epoch_loss)


for epoch in range(0, total_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    print('********************************************')
    print("training:")

    for img, mask in tqdm(data_loader_iter):
        it_train_num += 1
        solver.set_input(img, mask)
        train_loss = solver.combine_optimize()
        writer.add_scalar('dataset/train_loss%d' % epoch, train_loss,
                          it_train_num)
        train_epoch_loss += train_loss
        if it_train_num % 1000 == 0:
            valid(epoch, it_train_num)

    train_epoch_loss /= len(data_loader_iter)
    writer.add_scalar('dataset/train_epoch_loss', train_epoch_loss, epoch)

    valid(epoch, it_train_num)

    print('*********', file=mylog)
    print('epoch:', epoch, '    time:', int(time() - tic), file=mylog)
    print('train_loss:', train_epoch_loss, file=mylog)
    print('********************************************', file=mylog)
    print('epoch:', epoch, '    time:', int(time() - tic))
    print('train_loss:', train_epoch_loss)

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