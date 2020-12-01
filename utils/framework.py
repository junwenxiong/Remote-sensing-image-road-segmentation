import torch
import torch.nn as nn
from torch.autograd import Variable as V
from PIL import Image
import cv2 as cv
import numpy as np
from utils.Optimizer import Optim
from networks.Resnet18Unet import ResNet18Unet, ResNet34Unet, ResNeXt50Unet, ResNeXt50Unetv2
from networks.unet_model import Res34Unetv3, Res34Unetv4, Res34Unetv5, ResXt50Unetv5
from networks.dinknet import DinkNet50, DinkNet34, DinkNet50V2, DinkNet50V3_FCN
from networks.deeplab import DeepLabV3_FCN
from networks.CombineNet import CombineNet
from networks.baseline import UNet
from utils.loss_2 import SegmentationLosses
from utils.ranger import Ranger  # this is from ranger.py
from utils.loss import dice_bce_loss
from networks.Ensemble import MyEnsemble
# Mixed Precision training
from apex import amp
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel as DPP


class MyFrame():
    def __init__(self, args=None, evalmode=False):
        self.args = args

        if args.train == 'True':
            if args.backbone == 'unet':
                self.net = UNet()
            elif args.backbone == 'resunet34':
                self.net = ResNet34Unet(dyrelu=args.dyrelu)
            elif args.backbone == 'resunet18':
                self.net = ResNet18Unet()
            elif args.backbone == 'resxtunet50':
                self.net = ResNeXt50Unet()
            elif args.backbone == 'resnext50unetv2':
                self.net = ResNeXt50Unetv2()
            elif args.backbone == 'res34unetv5':
                self.net = Res34Unetv5(pretrained=True, dyrelu=args.dyrelu)
            elif args.backbone == 'resxt50unetv5':
                self.net = ResXt50Unetv5(pretrained=True)
            elif args.backbone == 'dinknet34':
                self.net = DinkNet34(pretrained=True)
            elif args.backbone == 'dinknet50v2':
                self.net = DinkNet50V2(pretrained=True)
            elif args.aux and args.backbone == 'dinknet50v3_fcn':
                self.net = DinkNet50V3_FCN(pretrained=True)
            elif args.aux and args.backbone == 'deeplabv3_fcn':
                self.net = DeepLabV3_FCN(pretrained=True)
                
        if args.combine == 'True':
            self.model1_name = args.model1
            self.model2_name = args.model2

            if self.model1_name == 'unet':
                self.modelA = UNet()
            elif self.model1_name == 'resunet34':
                self.modelA = ResNet34Unet(pretrained=False)
            elif self.model1_name == 'resunet18':
                self.modelA = ResNet18Unet(pretrained=False)
            elif self.model1_name == 'resxtunet34':
                self.modelA = ResNeXt50Unet(pretrained=False)
            elif self.model1_name == 'res34unetv5':
                self.modelA = Res34Unetv5(pretrained=False, dyrelu=False)
            elif self.model1_name == 'dinknet34':
                self.modelA = DinkNet34(pretrained=False)

            if self.model2_name == 'unet':
                self.modelB = Unet()
            elif self.model2_name == 'resunet34':
                self.modelB = ResNet34Unet(pretrained=False)
            elif self.model2_name == 'resunet18':
                self.modelB = ResNet18Unet(pretrained=False)
            elif self.model2_name == 'resxtunet34':
                self.modelB = ResNeXt50Unet(pretrained=False)
            elif self.model2_name == 'res34unetv5':
                self.modelB = Res34Unetv5(pretrained=False, dyrelu=False)
            elif self.model2_name == 'dinknet34':
                self.modelB = DinkNet34(pretrained=False)

            self.modelA.cuda()
            self.modelB.cuda()
            self.modelA_checkpoint = torch.load(args.modelA_checkpoint)
            self.modelB_checkpoint = torch.load(args.modelB_checkpoint)

            self.modelA.load_state_dict(self.modelA_checkpoint)
            self.modelB.load_state_dict(self.modelB_checkpoint)

            # Freeze these models
            for param in self.modelA.parameters():
                param.requires_grad_(False)
            for param in self.modelB.parameters():
                param.requires_grad_(False)

            self.net = MyEnsemble(2, 1)

        if args.test == 'True':
            if args.backbone == 'unet':
                self.net = UNet()
            elif args.backbone == 'resunet34':
                self.net = ResNet34Unet()
            elif args.backbone == 'resunet18':
                self.net = ResNet18Unet()
            elif args.backbone == 'resxtunet50':
                self.net = ResNeXt50Unet()
            elif args.backbone == 'resnext50unetv2':
                self.net = ResNeXt50Unetv2()
            elif args.backbone == 'res34unetv5':
                self.net = Res34Unetv5(pretrained=False, dyrelu=False)
            elif args.backbone == 'dinknet34':
                self.net = DinkNet34(pretrained=False)
            elif args.backbone == 'combinenet':
                self.net = CombineNet()

        self.old_lr = args.learn_rate

        # if train with mixed precision
        if not args.mixed_train:
            self.net = self.net.cuda()
            self.optimizer = Optim(self.net.parameters(),
                                   lr=args.learn_rate).build(args.optim)
            self.CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=150, eta_min=0)
            self.loss = SegmentationLosses(cuda=True).build_loss(args.loss)
            # self.net, self.optimizer = amp.initialize(self.net,
            #                                           self.optimizer,
            #                                           opt_level="O1")
        else:
            self.device = torch.device(f'cuda:{args.local_rank}')
            self.net = convert_syncbn_model(self.net).to(self.device)
            self.optimizer = Optim(self.net.parameters(),
                                   lr=args.learn_rate).build(args.optim)
            self.CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=150, eta_min=0)
            self.loss = SegmentationLosses(cuda=True).build_loss(args.loss)
            self.net, self.optimizer = amp.initialize(self.net,
                                                      self.optimizer,
                                                      opt_level="O1")
            self.net = DPP(self.net)

    # use this function if combine equals ture
    def set_combine_input(self, img_batch, mask_batch=None):
        self.combine_img = img_batch
        self.combine_mask = mask_batch

    def combine_val_optimize(self):
        self.forward()
        self.optimizer.zero_grad()

        final_pred = self.net(self.img)
        loss = self.loss(final_pred, self.mask)

        final_pred_label = final_pred.squeeze().cpu().data.numpy()
        gt = self.mask.squeeze().cpu().data.numpy()
        final_pred_label[final_pred_label > 0.5] = 1
        final_pred_label[final_pred_label <= 0.5] = 0

        confusion_matrix = self._generate_matrix(
            gt.astype(np.int8), final_pred_label.astype(np.int8))
        miou = self._Class_IOU(confusion_matrix)
        acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()

        return miou, acc, loss.item()

    def combine_optimize(self):
        self.forward()
        self.optimizer.zero_grad()

        final_pred = self.net.forward(self.img)

        loss = self.loss(
            final_pred,
            self.mask,
        )
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()

        self.optimizer.step()
        return loss.item()

    def ensemble_val_optimize(self):
        self.optimizer.zero_grad()
        modelA_pred = self.modelA_multi_scale_predict(self.combine_img)
        modelB_pred = self.modelB_multi_scale_predict(self.combine_img)

        net_input = torch.cat([modelA_pred, modelB_pred], 1)
        output = self.net.forward(net_input)
        loss = self.loss(output, self.combine_mask)
        pred_label = output.squeeze().cpu().data.numpy()
        gt = self.combine_mask.squeeze().cpu().data.numpy()
        pred_label[pred_label > 0.5] = 1
        pred_label[pred_label <= 0.5] = 0

        confusion_matrix = self._generate_matrix(gt.astype(np.int8),
                                                 pred_label.astype(np.int8))
        miou = self._Class_IOU(confusion_matrix)
        acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()

        return miou, acc, loss.item()

    def ensemble_optimize(self):
        self.optimizer.zero_grad()

        modelA_pred = self.modelA_multi_scale_predict(self.combine_img)
        modelB_pred = self.modelB_multi_scale_predict(self.combine_img)

        net_input = torch.cat([modelA_pred, modelB_pred], 1)
        output = self.net.forward(net_input)
        loss = self.loss(output, self.combine_mask)
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()

        self.optimizer.step()
        return loss.item()

    def modelA_multi_scale_predict(self, image_ori: Image):
        h, w = image_ori.shape[0], image_ori[1]
        self.modelA.eval()

        sample_ori = image_ori.copy()
        output_ori = self.modelA_predict(sample_ori)

        # rotate three angles for predicting
        angle_list = [90, 180, 270]
        for angle in angle_list:
            img_rotate = image_ori.rotate(angle, Image.BILINEAR)
            output = self.modelA_predict(img_rotate)
            pred = output.data.cpu().numpy()
            pred = pred.transpose((1, 2, 0))
            m_rotate = cv2.getRotationMatrix2D((h // 2, w // 2), 360.0 - angle,
                                               1)
            pred = cv2.warpAffine(pred, m_rotate, (h, w))
            pred = pred.transpose(2, 0, 1)
            output = torch.from_numpy(np.array([
                pred,
            ])).float()
            output_ori = torch.cat([output_ori, output.cuda()], 1)

        # vertical flip
        img_flip = image_ori.transpose(Image.FLIP_TOP_BOTTOM)
        output = self.modelA_predict(img_flip)
        pred = output.data.cpu().numpy()
        pred = pred.transpose((1, 2, 0))
        pred = cv2.flip(pred, 0)
        pred = pred.transpose((2, 0, 1))
        output = torch.from_numpy(np.array([
            pred,
        ])).float()
        output_ori = torch.cat([output_ori, output.cuda()], 1)

        # horizontal flip
        img_flip = image_ori.transpose(Image.FLIP_LEFT_RIGHT)
        output = self.modelA_predict(img_flip)
        pred = output.data.cpu().numpy()
        pred = pred.transpose((1, 2, 0))
        pred = cv2.flip(pred, 1)
        pred = pred.transpose((2, 0, 1))
        output = torch.from_numpy(np.array([
            pred,
        ])).float()
        output_ori = torch.cat([output_ori, output.cuda()], 1)

        return output_ori

    def modelA_predict(self, img):
        img = img.cuda()
        with torch.no_grad():
            output = self.modelA(img)
        return output

    def modelB_multi_scale_predict(self, image_ori: Image):
        h, w = image_ori.shape[0], image_ori[1]
        self.modelB.eval()

        sample_ori = image_ori.copy()
        output_ori = self.modelB_predict(sample_ori)

        # rotate three angles for predicting
        angle_list = [90, 180, 270]
        for angle in angle_list:
            img_rotate = image_ori.rotate(angle, Image.BILINEAR)
            output = self.modelB_predict(img_rotate)
            pred = output.data.cpu().numpy()
            pred = pred.transpose((1, 2, 0))
            m_rotate = cv2.getRotationMatrix2D((h // 2, w // 2), 360.0 - angle,
                                               1)
            pred = cv2.warpAffine(pred, m_rotate, (h, w))
            pred = pred.transpose(2, 0, 1)
            output = torch.from_numpy(np.array([
                pred,
            ])).float()
            output_ori = torch.cat([output_ori, output.cuda()], 1)

        # vertical flip
        img_flip = image_ori.transpose(Image.FLIP_TOP_BOTTOM)
        output = self.modelB_predict(img_flip)
        pred = output.data.cpu().numpy()
        pred = pred.transpose((1, 2, 0))
        pred = cv2.flip(pred, 0)
        pred = pred.transpose((2, 0, 1))
        output = torch.from_numpy(np.array([
            pred,
        ])).float()
        output_ori = torch.cat([output_ori, output.cuda()], 1)

        # horizontal flip
        img_flip = image_ori.transpose(Image.FLIP_LEFT_RIGHT)
        output = self.modelB_predict(img_flip)
        pred = output.data.cpu().numpy()
        pred = pred.transpose((1, 2, 0))
        pred = cv2.flip(pred, 1)
        pred = pred.transpose((2, 0, 1))
        output = torch.from_numpy(np.array([
            pred,
        ])).float()
        output_ori = torch.cat([output_ori, output.cuda()], 1)

        return output_ori

    def modelB_predict(self, img):
        img = img.cuda()
        with torch.no_grad():
            output = self.modelB(img)
        return output

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def test_one_img(self, img):
        pred = self.net.forward(img)

        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask

    def test_batch(self):
        self.forward(volatile=True)
        mask = self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return mask, self.img_id

    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())

        mask = self.net.forward(
            img).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return mask

    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        loss = self.loss(
            pred,
            self.mask,
        )  # pred mask : 20*1*1024*1024
        # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #     scaled_loss.backward()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # 验证时使用
    def _generate_matrix(self, gt_image, pre_image, num_class=2):
        mask = (gt_image >= 0) & (
            gt_image < num_class
        )  # ground truth中所有正确(值在[0, classe_num])的像素label的mask
        label = num_class * gt_image[mask].astype('int') + pre_image[mask]
        # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        count = np.bincount(label, minlength=num_class**2)
        confusion_matrix = count.reshape(num_class,
                                         num_class)  # 21 * 21(for pascal)
        return confusion_matrix

    def _Class_IOU(self, confusion_matrix):
        MIoU = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) +
                                            np.sum(confusion_matrix, axis=0) -
                                            np.diag(confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def val_optimize(self):
        self.forward()
        self.optimizer.zero_grad()

        pred = self.net.forward(self.img)
        loss = self.loss(
            pred,
            self.mask,
        )
        if self.args.aux:
            pred = pred[0]
        pred_label = pred.squeeze().cpu().data.numpy()
        gt = self.mask.squeeze().cpu().data.numpy()
        pred_label[pred_label > 0.5] = 1
        pred_label[pred_label <= 0.5] = 0
        #mask = pred_label*255
        #cv2.imwrite('mask1.png', mask.astype(np.uint8))
        confusion_matrix = self._generate_matrix(gt.astype(np.int8),
                                                 pred_label.astype(np.int8))
        miou = self._Class_IOU(confusion_matrix)
        acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()

        return miou, acc, loss.item()

    def base_optimize(self):

        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)

        loss = self.loss(self.mask, pred)  # pred mask : 20*1*1024*1024
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def combine_save(self, path):
        weight_dict = {
            'model1': self.model1.state_dict(),
            'model2': self.model2.state_dict(),
            'combinenet': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(weight_dict, path)

    def save(self, path):
        # wraped the module layer when use the mixed precision
        self.model_state_dict = self.net.module.state_dict() if len(
            self.args.gpu_ids) > 1 else self.net.state_dict()

        weight_dict = {
            'model': self.model_state_dict,
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(weight_dict, path)

    def combine_load(self, path):
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['combinenet'])
        self.model1.load_state_dict(checkpoint['model1'])
        self.model2.load_state_dict(checkpoint['model2'])

    def load2(self, path):
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['model'])

    def load(self, path):
        self.net.load_state_dict(torch.load(path)['model'])

    def update_lr(self, new_lr, mylog, factor=False):
        return self.CosineLR.step()

    def save_img(pred, src):
        img = np.float32(pred)
        cv.cvtColor(pred, cv.COLOR_GRAY2BGR)
        cv.imwrite(src, img)