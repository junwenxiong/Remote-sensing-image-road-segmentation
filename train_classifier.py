import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from utils.data import ImageFolder, ImageFolderv2
from classifier.model import XNet
import argparse
import os
from tqdm import tqdm
from time import time, strftime, localtime

def train(args, model, train_dataloders, val_dataloaders, criterion, optimizer,
          scheduler, num_epochs, dataset_sizes):

    now = strftime('%m-%d-%H-%M', localtime())
    mylog = open('logs/' + 'resnet34' + now + '.log', 'w')
    print(args, file=mylog)

    checkpoint_dir = 'weights/resnet34/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    best_acc = 0
    for epoch in range(0, num_epochs):
        model.train()
        train_loss = 0
        train_corrects = 0
        for i, (inputs, labels) in enumerate(train_dataloders):
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels.long())

            loss.backward()
            optimizer.step()

            # import pdb
            # pdb.set_trace()
            _, predicted = torch.max(outputs.data, 1)
            train_loss += loss.item()
            train_corrects += torch.sum(predicted == labels.data)

            batch_loss = torch.true_divide(train_loss,
                                           ((i + 1) * args.batch_size))
            batch_acc = torch.true_divide(train_corrects,
                                          ((i + 1) * args.batch_size))

            if i % 500 == 0:
                print(
                    '[Epoch {}/{}]-[batch:{}/{}] train Loss: {:.6} Acc:{:.4f}'.
                    format(epoch, num_epochs - 1, i,
                           round(dataset_sizes / args.batch_size) - 1,
                           batch_acc, batch_acc))
        epoch_loss = torch.true_divide(train_loss, dataset_sizes)
        epoch_acc = torch.true_divide(train_corrects, dataset_sizes)

        print('epoch: {} Loss: {:.4f}, Acc:{:.4f}'.format(
            epoch, epoch_loss, epoch_acc))

        model.eval()
        val_loss = 0
        val_corrects = 0
        val_dataset_sizes = len(val_dataloaders)
        for i, (inputs, labels) in enumerate(val_dataloaders):
            inputs = inputs.cuda()
            labels = labels.cuda()

            with torch.no_grad():
                outputs = model(inputs)

            loss = criterion(outputs, labels.long())

            _, predicted = torch.max(outputs.data, 1)

            val_loss += loss.item()
            val_corrects += torch.sum(predicted == labels.data)

            batch_loss = torch.true_divide(val_loss,
                                           ((i + 1) * args.batch_size))
            batch_acc = torch.true_divide(val_corrects,
                                          ((i + 1) * args.batch_size))

            if i % 500 == 0:
                print(
                    '[Eopoch {}/{}]-[batch:{}/{}] val Loss: {:.6f} Acc: {:.4f}'
                    .format(epoch, num_epochs - 1, i,
                            round(dataset_sizes / args.batch_size) - 1,
                            batch_loss, batch_acc), file=mylog)
                print(
                    '[Eopoch {}/{}]-[batch:{}/{}] val Loss: {:.6f} Acc: {:.4f}'
                    .format(epoch, num_epochs - 1, i,
                            round(dataset_sizes / args.batch_size) - 1,
                            batch_loss, batch_acc))

        epoch_loss = torch.true_divide(val_loss, val_dataset_sizes)
        epoch_acc = torch.true_divide(val_corrects, val_dataset_sizes)

        print('epoch: {} Loss: {:.4f}, Acc:{:.4f}'.format(
            epoch, epoch_loss, epoch_acc), file=mylog)
        print('epoch: {} Loss: {:.4f}, Acc:{:.4f}'.format(
            epoch, epoch_loss, epoch_acc))

        mylog.flush()

        if epoch_acc > best_acc:
            weight_dict = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            best_acc = epoch_acc
            torch.save(weight_dict,
                       checkpoint_dir + 'resnet34_{}.pth'.format(epoch))
                       
        scheduler.step()
    mylog.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of XNet")
    parser.add_argument('--dataset', type=str, default="./huawei")
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--num-class', type=int, default=2)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.045)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--gpus', type=str, default='1')
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--save-epoch-freq', type=int, default=1)
    parser.add_argument('--save-path', type=str, default="output")
    parser.add_argument('--resume',
                        type=str,
                        default="",
                        help="For training from one checkpoint")
    parser.add_argument('--start-epoch',
                        type=int,
                        default=0,
                        help="Corresponding to the epoch of resume ")
    parser.add_argument('--network', type=str, default="se_resnet_50", help="")
    args = parser.parse_args()

    # 训练测试 256*256
    train_ROOT = os.path.join(args.dataset, 'train')
    val_ROOT = os.path.join(args.dataset, 'val')

    train_image_dir = os.path.join(train_ROOT, 'images')
    val_image_dir = os.path.join(val_ROOT, 'images')

    train_list = os.listdir(train_image_dir)
    val_list = os.listdir(val_image_dir)

    train_csv = 'preprocess/train_label.csv'
    val_csv = 'preprocess/val_label.csv'

  
    dataset = ImageFolderv2(train_ROOT, train_csv, mode='train')
    train_dataloders = torch.utils.data.DataLoader(dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers)

    val_dataset = ImageFolderv2(val_ROOT, val_csv, mode='val')
    val_dataloaders = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=args.workers)
    dataset_sizes = dataset.__len__()
    model = XNet().cuda()

    checkpoint = torch.load('./weights/resnet34/resnet34_1.pth')
    model.load_state_dict(checkpoint['state_dict'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=0.00004)
    optimizer.load_state_dict(checkpoint['optimizer'])
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)

    train(args=args,
          model=model,
          train_dataloders=train_dataloders,
          val_dataloaders=val_dataloaders,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=exp_lr_scheduler,
          num_epochs=args.num_epochs,
          dataset_sizes=dataset_sizes)
