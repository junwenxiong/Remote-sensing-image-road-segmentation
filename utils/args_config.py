import argparse


def make_args():
    parser = argparse.ArgumentParser(description="Pytorch Unet Training")
    parser.add_argument('--backbone',
                        type=str,
                        default='unet',
                        choices=[
                            'unet', 'resunet34', 'resunet18', 'resxtunet34',
                            'combinenet', 'res34unetv5', 'dinknet34',
                            'resnext50unetv2', 'dinknet50v2'
                        ],
                        help="backbone name (default: unet)")
    parser.add_argument('--dyrelu',
                        type=str,
                        default=False,
                        help="decide whether to use dynamic relu func ")
    parser.add_argument('--dataset',
                        type=str,
                        default='./huawei/',
                        help="dataset dir ")
    parser.add_argument('--workers',
                        type=int,
                        default=8,
                        metavar='N',
                        help='dataloader threads')
    parser.add_argument('--local_rank',
                        default=-1,
                        type=int,
                        help='ranking within the nodes')
    parser.add_argument('--gpu-ids',
                        type=str,
                        default='0',
                        help='use which gpu to train')

    # training hyperparameters
    parser.add_argument('--epochs',
                        type=int,
                        default=40,
                        metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--batch-size',
                        type=int,
                        default=16,
                        metavar='N',
                        help='input batch size for train ')
    parser.add_argument('--learn-rate',
                        type=float,
                        default=1e-4,
                        metavar='LR',
                        help='learning rate')
    parser.add_argument('--optim',
                        default='sgd',
                        type=str,
                        choices=['adam', 'sgd', 'rmsprop', 'ranger'],
                        help='optimizer type ')
    parser.add_argument('--loss',
                        default='lovasz',
                        type=str,
                        choices=[
                            'dice_bce', 'lovasz', 'ce', 'focal', 'dice',
                            'mixed', 'dice_bce_focal'
                        ],
                        help='loss type ')
    parser.add_argument('--resume',
                        type=str,
                        default=None,
                        help='put the path to resuming file if needed')

    # train mode
    parser.add_argument(
        '--combine',
        type=str,
        default=False,
        help='decide to whether to combine model given by checkpoint')
    parser.add_argument(
        '--model1',
        type=str,
        default=None,
        help='choose this model2 to fuse with model2 if combine is true')
    parser.add_argument('--model1-checkpoint',
                        type=str,
                        default=None,
                        help='use this model1_checkpoint  if combine is true')
    parser.add_argument(
        '--model2',
        type=str,
        default=None,
        help='choose this model2 to fuse with model1 if combine is true')
    parser.add_argument('--model2-checkpoint',
                        type=str,
                        default=None,
                        help='use this model2_checkpoint  if combine is true')
    parser.add_argument('--test',
                        type=str,
                        default=False,
                        help='dive into test mode')
    parser.add_argument('--train',
                        type=str,
                        default=True,
                        help='dive into train mode')
    parser.add_argument('--mixed-train',
                        type=str,
                        default=False,
                        help='dive into mixed train mode')

    parser.add_argument('--iter-num', type=int, help='iter num')
    args = parser.parse_args()
    return args
