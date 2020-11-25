from networks.Resnet18Unet import ResNet18Unet, ResNet34Unet, ResNeXt50Unet
from networks.unet_att_Dyrelu import UNet_att_Dyre
from networks.dlinknet import DLinkNet50
from networks.unet_model import Res34Unetv3, Res34Unetv5
from networks.unet import Unet
import torch
from torchstat import stat

def print_model_info(model):
    stat(model, (3,256, 256))
    

if __name__ == "__main__":
    from torch.autograd import Variable
    x = Variable(torch.randn(1,3, 256, 256))
    res34unet = ResNet34Unet()
    resxt = ResNeXt50Unet()
    res34 = Res34Unetv3()
    res34v5 = Res34Unetv5(dyrelu=True)

    y = resxt(x)
    print(y.shape)
    print_model_info(resxt)