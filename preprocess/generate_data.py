import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import random
from glob import glob
import shutil
from random import shuffle
Image.MAX_IMAGE_PIXELS = 1000000000000000
cv2.CV_IO_MAX_IMAGE_PIXELS = 1000000000000000
#TARGET_W, TARGET_H = 1024, 1024
os.environ.setdefault('OPENCV_IO_MAX_IMAGE_PIXELS', '1000000000000000')
#TARGET_W, TARGET_H = 256, 256
#TARGET_W, TARGET_H = 1024, 1024

img_w = 384
img_h = 384
#import thread
from multiprocessing import Pool, Process

import time
image_sets = ['182.png', '382.png']
label_sets = ['182_label.png', '382_label.png']


def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb, yb


def blur(img):
    img = cv2.blur(img, (3, 3))
    return img


def add_noise(img):
    for i in range(200):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img


def data_augment(xb, yb, only_rotate):
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 90)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 180)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)
    if only_rotate:
        return xb, yb

    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb, 1.0)

    if np.random.random() < 0.25:
        xb = blur(xb)

    if np.random.random() < 0.2:
        xb = add_noise(xb)

    return xb, yb


# 在原图上随机裁剪
def creat_dataset(image_num, mode='augment'):
    print('creating dataset...')

    target_w, target_h = TARGET_W, TARGET_H
    overlap = target_h // 8
    stride = target_h - overlap
    image_path = "/data/home/zy/huawei/182.png"
    label_path = "/data/home/zy/huawei/182_label.png"
    image = np.asarray(Image.open(image_path))
    label = np.asarray(Image.open(label_path))
    h, w = image.shape[0], image.shape[1]
    print("原始大小: ", w, h)
    if (w - target_w) % stride:
        new_w = w
    if (h - target_h) % stride:
        new_h = h
    src_img = cv2.copyMakeBorder(image, 0, new_h - h, 0, new_w - w,
                                 cv2.BORDER_CONSTANT, 0)
    label_img = cv2.copyMakeBorder(label, 0, new_h - h, 0, new_w - w,
                                   cv2.BORDER_CONSTANT, 1)
    h, w = src_img.shape[0], src_img.shape[1]
    print("填充至整数倍: ", w, h)

    if src_img.shape[0] != label_img.shape[0]:
        print("Error!!!!!!!!!!!!")

    X_height, X_width = src_img.shape[0], src_img.shape[1]
    count = 0
    while count < 20:
        random_width = random.randint(0, X_width - img_w - 1)
        random_height = random.randint(0, X_height - img_h - 1)
        src_roi = src_img[random_height:random_height + img_h,
                          random_width:random_width + img_w]
        label_roi = label_img[random_height:random_height + img_h,
                              random_width:random_width + img_w]
        if src_roi.max() < 0.1:
            continue

        # if mode == 'augment':
        #     src_roi, label_roi = data_augment(src_roi, label_roi)
        #img_path =  '/data/home/zy/huawei/data_1024_augment/train/labels/'
        cv2.imwrite(
            ('/data/home/zy/huawei/data_1024_augment/test/images/%d.png' %
             count), src_roi)
        cv2.imwrite(
            ('/data/home/zy/huawei/data_1024_augment/test/labels/%d.png' %
             count), label_roi)
        print("success%d" % count)
        count += 1
        g_count += 1


# 删除纯黑色的图片
def delete_data():
    image_dir = "/data/home/zy/huawei/new_data/512/val/images/"
    label_dir = "/data/home/zy/huawei/new_data/512/val/labels/"
    img_file = glob(image_dir + '*' + '.png')

    for img_path in img_file:
        img_name = img_path.split(os.sep)[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]

        label_path = label_dir + imidx + '.png'
        img = Image.open(img_path)
        img = np.array(img)
        if img.max() < 0.1:
            os.remove(img_path)
            os.remove(label_path)
            print("Success remove")
    image_nums = len(os.listdir(image_dir))
    val_nums = len(os.listdir(label_dir))

    print("image: " + str(image_nums))
    print("val: " + str(val_nums))


def cut_data():
    val_ROOT = '/data/home/zy/huawei/berlin/'
    val_image_dir = '/data/home/zy/huawei/berlin/images/'
    val_label_dir = '/data/home/zy/huawei/berlin/labels/'
    img_file = glob(val_image_dir + '*' + '.png')
    g_count = 0
    for img_path in img_file:
        img_name = img_path.split(os.sep)[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]

        label_path = val_label_dir + imidx + '.png'
        img = cv2.imread(img_path)
        label = cv2.imread(label_path)
        #tmp = img.size
        X_height, X_width = img.shape[0], img.shape[1]
        random_width = 0
        random_height = 0
        for i in range(0, 3):
            src_roi = img[random_height:random_height + 256,
                          random_width:random_width + 256]
            label_roi = label[random_height:random_height + 256,
                              random_width:random_width + 256]

            if not os.path.exists('/data/home/zy/huawei/berlin_256/images/'):
                os.mkdir('/data/home/zy/huawei/berlin_256/images/')
            if not os.path.exists('/data/home/zy/huawei/berlin_256/labels/'):
                os.mkdir('/data/home/zy/huawei/berlin_256/labels/')

            cv2.imwrite(
                ('/data/home/zy/huawei/berlin_256/images/%d.png' % g_count),
                src_roi)
            cv2.imwrite(
                ('/data/home/zy/huawei/berlin_256/labels/%d.png' % g_count),
                label_roi)
            random_height += 256
            random_width += 256
            g_count += 1


data_dir = "/data/home/zy/huawei/new_data/512/train/"
train_dir = data_dir + "images/"
label_dir = data_dir + "labels/"

img_file = glob(train_dir + '*' + '.png')

data_size = len(img_file)
print("原本数据集大小：", data_size)
g_count = 0
new_datasize = 30000
count = new_datasize - data_size


def creat_data(data_dir, only_rotate=False):
    print('creating dataset...')

    # for img_path in img_file:
    #     src_img = cv2.imread(img_path)
    #     if src_img.shape[0] == 1024:
    #         count = count + 1

    #for g_count in tqdm(range(0, 30000)):
    g_count = 0
    while g_count < count:

        indix = random.randint(0, data_size - 1)
        img_path = img_file[indix]
        img_name = img_path.split(os.sep)[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]

        label_path = label_dir + imidx + '.png'
        #label_path = '/data/home/zy/huawei/new_data/1024/train/labels/233.png'
        if not os.path.exists(label_path):
            os.remove(img_path)
            continue
        src_img = cv2.imread(img_path)  # 3 channels
        label_img = cv2.imread(label_path,
                               cv2.IMREAD_GRAYSCALE)  # single channel

        if src_img.shape[0] != label_img.shape[0]:
            continue
        #print(src_img.shape[0], src_img.shape[1])
        src_roi, label_roi = data_augment(src_img, label_img, only_rotate)
        name = '%d_%d.png' % (g_count, os.getpid())

        cv2.imwrite((train_dir + name), src_roi)
        cv2.imwrite((label_dir + name), label_roi)
        g_count += 1
        if src_img.shape[0] != 512:
            break
        #print(src_roi.shape[0], src_roi.shape[1])
        print(name)


# 数据检查
def test_img(data_dir):
    print('creating dataset...')

    train_dir = data_dir + "images/"
    label_dir = data_dir + "labels/"

    img_file = glob(train_dir + '*' + '.png')
    shuffle(img_file)
    data_size = len(img_file)
    print("原本数据集大小：", data_size)
    g_count = 0

    count = 0
    result = []
    for img_path in img_file:
        # src_img = cv2.imread(img_path)
        # if src_img.shape[0] == 1024:
        #     count = count + 1

        img_name = img_path.split(os.sep)[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]

        label_path = label_dir + imidx + '.png'
        src_img = cv2.imread(img_path)  # 3 channels
        # label_img = cv2.imread(label_path,
        #                        cv2.IMREAD_GRAYSCALE)  # single channel
        # if not os.path.exists(label_path):
        #     # os.remove(os.path.join(train_dir, img_name))
        #     result.append(label_path)

        # if src_img.shape[0] != label_img.shape[0] or src_img.shape[
        #         1] != label_img.shape[1]:
        #     print(src_img.shape[0])
        #     os.remove(img_path)
        #     os.remove(label_path)
        print(img_path)
        src_img = np.array(src_img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6

        # if src_img.shape != (384, 384, 3):
        #     result.append(img_path)

    print(count)
    print(result)


if __name__ == "__main__":
    arg = 1
    start = time.time()
    data_dir = "/data/home/zy/huawei/new_data/384/train/"

    #data_dir = "/data/home/zy/huawei/new_data/1024/val/"

    pool = Pool(processes=6)
    only_rotate = False
    #x_y = zip (data_dir, only_rotate)
    test_file = '/data/home/zy/huawei/new_data/384/train/images/04.png'

    # img = cv2.imread(test_file)
    # print(img.shape)
    test_img(data_dir)
    # pool.map(creat_data, data_dir)

    # with Pool(6) as p:
    #    p.map(creat_data, [data_dir,False])

    # p = Process(target=creat_data, args=(data_dir, False))
    # p.start()
    # p.join()

    #creat_data(data_dir,False)
    #test_img(data_dir)
    #

    #creat_data(data_dir)

    #delete_data()
    # train_dir = "/data/home/zy/huawei/train/images/"
    # train_nums = len(os.listdir(train_dir))
    # creat_dataset(image_num=100000-train_nums, mode='augment')
    # cut_data()