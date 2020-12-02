import torch
from tqdm import tqdm
import cv2
import os
from time import time, strftime, localtime
from tensorboardX import SummaryWriter
from utils.framework import MyFrame
from utils.data import ImageFolder, make_dataloader
from utils.args_config import make_args
from utils.loss_2 import SegmentationLosses
from torch.utils.data import DataLoader

args = make_args()
print(args)

NAME = args.backbone
now = strftime('-%m-%d-%H-%M', localtime())
model_dir = './weights/' + NAME + now + '/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

solver = MyFrame(args, )

train_dataloader, val_dataloader, _ = make_dataloader(args)

mylog = open('logs/' + NAME + '.log', 'w')
print(args, file=mylog)
mylog.flush()
tic = time()
no_optim = 0
total_epoch = args.epochs
train_epoch_best_loss = 100.

logdir = 'logs/' + NAME
writer = SummaryWriter(logdir, )


def valid(epoch, it_train_num):
    # *****验证*******
    val_data_loader_iter = iter(val_dataloader)
    val_epoch_loss = 0
    val_miou = 0.0
    val_acc = 0.0
    it_val_num = 0
    print("validing:")
    for img, mask in tqdm(val_data_loader_iter):
        it_val_num += 1
        solver.set_input(img, mask)
        miou, acc, val_loss = solver.val_optimize()
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

    return val_miou


best_miou = 0
for epoch in range(0, total_epoch + 1):
    data_loader_iter = iter(train_dataloader)
    train_epoch_loss = 0
    print('********************************************')
    print("training:")
    it_train_num = 0
    for img, mask in tqdm(data_loader_iter):
        it_train_num += 1
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        writer.add_scalar('dataset/train_loss%d' % epoch, train_loss,
                          it_train_num)
        train_epoch_loss += train_loss

    train_epoch_loss /= len(data_loader_iter)
    writer.add_scalar('dataset/train_epoch_loss', train_epoch_loss, epoch)

    val_miou = valid(epoch, it_train_num)

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
        if best_miou < val_miou:
            best_miou = val_miou
            solver.save(model_dir + NAME +
                        '{}_miou_{}.pth'.format(epoch, val_miou))
            str = NAME + '{}_miou_{}.pth'.format(epoch, val_miou)
            
    if no_optim > 8:
        print('early stop at %d epoch', file=mylog)

        print('early stop at %d epoch' % epoch)
        break
    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        solver.load(model_dir + str)
        solver.update_lr(5.0, factor=True, mylog=mylog)

    mylog.flush()

print('Finish!', file=mylog)
print('Finish!')
mylog.close()