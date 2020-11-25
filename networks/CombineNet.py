import torch.nn as nn
import torch.nn.functional as F

# initialize the module
def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight)


class CombineNet(nn.Module):
    def __init__(self, in_chnnels=2, n_classes=1):
        super(CombineNet, self).__init__()

        self.conv1 = nn.Conv2d(in_chnnels, n_classes * 3, 3, 1, 1)
        self.conv2 = nn.Conv2d(n_classes * 3, n_classes, 3, 1, 1)

        for m in self.modules():
            init_weights(m, init_type='kaiming')
    
    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)

        return F.sigmoid(conv2)
