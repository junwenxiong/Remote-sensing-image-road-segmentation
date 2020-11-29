"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import os
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from albumentations import (PadIfNeeded, HorizontalFlip, VerticalFlip,
                            CenterCrop, Crop, Compose, Transpose,
                            RandomRotate90, ElasticTransform, GridDistortion,
                            OpticalDistortion, RandomSizedCrop, OneOf, CLAHE,
                            RandomContrast, RandomGamma, RandomBrightness)

u2 = 0.15  # 默认0.5


def randomHueSaturationValue(image,
                             hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255),
                             u=u2):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0],
                                      hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image,
                           mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT,
                           u=u2):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect**0.5)
        sy = scale / (aspect**0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array(
            [width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image,
                                    mat, (width, height),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=borderMode,
                                    borderValue=(
                                        0,
                                        0,
                                        0,
                                    ))
        mask = cv2.warpPerspective(mask,
                                   mat, (width, height),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=borderMode,
                                   borderValue=(
                                       0,
                                       0,
                                       0,
                                   ))

    return image, mask


def randomHorizontalFlip(image, mask, u=u2):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def randomVerticleFlip(image, mask, u=u2):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask


def randomRotate90(image, mask, u=u2):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask


def transposeXY(image, mask):
    aug = Transpose(p=1)
    augmented = aug(image=image, mask=mask)
    image_transposed = augmented['image']
    mask_transposed = augmented['mask']

    return image_transposed, mask_transposed


def default_loader(id, root, mode):
    img = cv2.imread(os.path.join(root, 'images', '{}').format(id))
    mask = cv2.imread(
        os.path.join(root, 'labels', '{}').format(id), cv2.IMREAD_GRAYSCALE)

    mask = mask * 255
    if mode == 'train':
        img = randomHueSaturationValue(img,
                                       hue_shift_limit=(-30, 30),
                                       sat_shift_limit=(-5, 5),
                                       val_shift_limit=(-15, 15))

        img, mask = randomShiftScaleRotate(img,
                                           mask,
                                           shift_limit=(-0.1, 0.1),
                                           scale_limit=(-0.1, 0.1),
                                           aspect_limit=(-0.1, 0.1),
                                           rotate_limit=(-0, 0))
        img, mask = randomHorizontalFlip(img, mask)
        img, mask = randomVerticleFlip(img, mask)
        img, mask = randomRotate90(img, mask)
        img, mask = transposeXY(img, mask)

    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    return img, mask


def load_img_wo_label(id, root, mode):
    img = cv2.imread(os.path.join(root, 'images', '{}').format(id))
    if mode == 'train':
        img = randomHueSaturationValue(img,
                                       hue_shift_limit=(-30, 30),
                                       sat_shift_limit=(-5, 5),
                                       val_shift_limit=(-15, 15))

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    return img


class ImageFolder(data.Dataset):
    def __init__(self, trainlist, root, mode='train'):
        self.ids = trainlist
        self.loader = default_loader
        self.root = root
        self.mode = mode

    def __getitem__(self, index):
        id = self.ids[index]
        img, mask = self.loader(id, self.root, mode=self.mode)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        return len(self.ids)


class ImageFolderv2(data.Dataset):
    def __init__(self, root, labels, mode='train'):
        self.loader = load_img_wo_label
        self.root = root
        self.label = labels
        self.mode = mode

    def __getitem__(self, index):
        id_image, id_label = self.read_label(index, self.label)
        img = self.loader(id_image, self.root, self.mode)
        img = torch.Tensor(img)
        return img, id_label

    def __len__(self):
        return len(pd.read_csv(self.label))

    def read_label(self, id, labels):
        labels_df = pd.read_csv(labels)
        id_image = labels_df['fname'][id]
        id_label = labels_df['street'][id]

        return id_image, id_label


def make_dataloader(args):
    train_ROOT = os.path.join(args.dataset, 'train')
    val_ROOT = os.path.join(args.dataset, 'val')
    test_ROOT = os.path.join(args.dataset, 'test')

    train_image_dir = os.path.join(train_ROOT, 'images')
    val_image_dir = os.path.join(val_ROOT, 'images')
    train_list = os.listdir(train_image_dir)
    val_list = os.listdir(val_image_dir)
    test_image_dir = os.path.join(test_ROOT, 'images')
    test_list = os.listdir(test_image_dir)
    # import pdb
    # pdb.set_trace()
    train_dataset = ImageFolder(train_list, train_ROOT, mode='train')
    val_dataset = ImageFolder(val_list, val_ROOT, mode='val')
    test_dataset = ImageFolder(test_list, test_ROOT, mode='val')
    train_data_loader, val_data_loader, test_data_loader = None, None, None
    shuffle = True
    train_sampler = None
    val_sampler = None
    if args.train:
        if args.mixed_train:
            train_sampler = DistributedSampler(train_dataset)
            val_sampler = DistributedSampler(val_dataset)
            shuffle = False

        train_data_loader = DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=shuffle,
                                       num_workers=args.workers,
                                       sampler=train_sampler)

        val_data_loader = DataLoader(val_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=shuffle,
                                     num_workers=args.workers,
                                     sampler=val_sampler)
    else:
        test_data_loader = DataLoader(test_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.workers)

    return train_data_loader, val_data_loader, test_data_loader