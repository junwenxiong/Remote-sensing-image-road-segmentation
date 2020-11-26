"""
此代码将给定的两张图片及其标签切分成1024*1024的小图，步长选手可自行调整
随后会随机分成训练集和验证集，比例选手亦可随机调整
"""
import os
import numpy as np
from PIL import Image
import cv2 as cv
from tqdm import tqdm
import random
import shutil
Image.MAX_IMAGE_PIXELS = 1000000000000000
TARGET_W, TARGET_H = 384, 384
# TARGET_W, TARGET_H = 768, 768
# TARGET_W, TARGET_H = 256, 256
STEP = 992


# 切成9*9
def cut_images_9_9(image_name,
                   image_path,
                   label_path,
                   save_dir,
                   is_show=False):
    # 初始化路径
    image_save_dir = os.path.join(save_dir, "images/")
    if not os.path.exists(image_save_dir): os.makedirs(image_save_dir)
    label_save_dir = os.path.join(save_dir, "labels/")
    if not os.path.exists(label_save_dir): os.makedirs(label_save_dir)
    if is_show:
        label_show_save_dir = os.path.join(save_dir, "labels_show/")
        if not os.path.exists(label_show_save_dir):
            os.makedirs(label_show_save_dir)

    target_w, target_h = TARGET_W, TARGET_H

    # target_w, target_h = new_h//3 , new_w//3
    # overlap = target_h // 8
    # overlap = 0
    # stride = target_h - overlap

    image = np.asarray(Image.open(image_path))
    label = np.asarray(Image.open(label_path))
    h, w = image.shape[0], image.shape[1]
    target_w, target_h = (w // 3), (h // 3)
    overlap = target_h // 8
    overlap = 0
    stride_h = target_h - overlap
    stride_w = target_w
    print("原始大小: ", w, h)
    # if (w-target_w) % stride:
    #     new_w = ((w-target_w)//stride + 1)*stride + target_w
    # if (h-target_h) % stride:
    #     new_h = ((h-target_h)//stride + 1)*stride + target_h

    #image = cv.copyMakeBorder(image,0,new_h-h,0,new_w-w,cv.BORDER_CONSTANT,0)
    #label = cv.copyMakeBorder(label,0,new_h-h,0,new_w-w,cv.BORDER_CONSTANT,1)
    h, w = image.shape[0], image.shape[1]
    print("填充至整数倍: ", w, h)

    def crop(cnt, crop_image, crop_label, is_show=is_show):
        _name = image_name.split(".")[0]
        image_save_path = os.path.join(
            image_save_dir,
            _name + "_" + str(cnt[0]) + "_" + str(cnt[1]) + ".png")
        label_save_path = os.path.join(
            label_save_dir,
            _name + "_" + str(cnt[0]) + "_" + str(cnt[1]) + ".png")
        #label_show_save_path = os.path.join(label_show_save_dir, _name+"_"+str(cnt[0])+str(cnt[1])+".png")
        cv.imwrite(image_save_path, crop_image)
        cv.imwrite(label_save_path, crop_label)
        #if is_show:
        #    cv.imwrite(label_show_save_path, crop_label*255)

    h, w = image.shape[0], image.shape[1]
    # for i in tqdm(range((w-target_w)//stride + 1)):
    #     for j in range((h-target_h)//stride + 1):
    for i in tqdm(range(3)):
        for j in range(3):
            topleft_x = i * target_w
            topleft_y = j * target_h
            crop_image = image[topleft_y:topleft_y + target_h,
                               topleft_x:topleft_x + target_w]
            crop_label = label[topleft_y:topleft_y + target_h,
                               topleft_x:topleft_x + target_w]

            crop((i, j), crop_image, crop_label)


def cut_images(image_name, image_path, label_path, save_dir, is_show=False):
    # 初始化路径
    image_save_dir = os.path.join(save_dir, "images/")
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)
    label_save_dir = os.path.join(save_dir, "labels/")
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)

    if is_show:
        label_show_save_dir = os.path.join(save_dir, "labels_show/")
        if not os.path.exists(label_show_save_dir):
            os.makedirs(label_show_save_dir)

    target_w, target_h = TARGET_W, TARGET_H

    overlap = target_h // 3
    #overlap = 0
    stride = target_h - overlap

    image = np.asarray(Image.open(image_path))
    label = np.asarray(Image.open(label_path))
    h, w = image.shape[0], image.shape[1]
    print("原始大小: ", w, h)
    if (w - target_w) % stride:
        new_w = ((w - target_w) // stride + 1) * stride + target_w
    if (h - target_h) % stride:
        new_h = ((h - target_h) // stride + 1) * stride + target_h
    image = cv.copyMakeBorder(image, 0, new_h - h, 0, new_w - w,
                              cv.BORDER_CONSTANT, 0)
    label = cv.copyMakeBorder(label, 0, new_h - h, 0, new_w - w,
                              cv.BORDER_CONSTANT, 1)
    h, w = image.shape[0], image.shape[1]
    print("填充至整数倍: ", w, h)

    def crop(cnt, crop_image, crop_label, is_show=is_show):
        _name = image_name.split(".")[0]
        image_save_path = os.path.join(
            image_save_dir,
            _name + "_" + str(cnt[0]) + "_" + str(cnt[1]) + ".png")
        label_save_path = os.path.join(
            label_save_dir,
            _name + "_" + str(cnt[0]) + "_" + str(cnt[1]) + ".png")

        # label_show_save_path = os.path.join(label_show_save_dir, _name+"_"+str(cnt[0])+str(cnt[1])+".png")

        if crop_image.max() < 0.1:
            return
        cv.imwrite(image_save_path, crop_image)
        cv.imwrite(label_save_path, crop_label)
        if is_show:
            cv.imwrite(label_show_save_path, crop_label * 255)

    h, w = image.shape[0], image.shape[1]
    for i in tqdm(range((w - target_w) // stride + 1)):
        for j in range((h - target_h) // stride + 1):
            topleft_x = i * stride
            topleft_y = j * stride
            crop_image = image[topleft_y:topleft_y + target_h,
                               topleft_x:topleft_x + target_w]
            crop_label = label[topleft_y:topleft_y + target_h,
                               topleft_x:topleft_x + target_w]
            crop((i, j), crop_image, crop_label)


def get_train_val(data_dir):
    all_images_dir = os.path.join(data_dir, "images/")
    all_labels_dir = os.path.join(data_dir, "labels/")
    train_imgs_dir = os.path.join(data_dir, "train/images/")
    if not os.path.exists(train_imgs_dir):
        os.makedirs(train_imgs_dir)
    val_imgs_dir = os.path.join(data_dir, "val/images/")
    if not os.path.exists(val_imgs_dir):
        os.makedirs(val_imgs_dir)
    train_labels_dir = os.path.join(data_dir, "train/labels/")
    if not os.path.exists(train_labels_dir):
        os.makedirs(train_labels_dir)
    val_labels_dir = os.path.join(data_dir, "val/labels/")
    if not os.path.exists(val_labels_dir):
        os.makedirs(val_labels_dir)
    for name in os.listdir(all_images_dir):
        image_path = os.path.join(all_images_dir, name)
        label_path = os.path.join(all_labels_dir, name)
        if random.randint(0, 10) < 2:
            image_save = os.path.join(val_imgs_dir, name)
            label_save = os.path.join(val_labels_dir, name)
        else:
            image_save = os.path.join(train_imgs_dir, name)
            label_save = os.path.join(train_labels_dir, name)
        shutil.move(image_path, image_save)
        shutil.move(label_path, label_save)
    total_nums = len(os.listdir(all_images_dir))
    train_nums = len(os.listdir(train_imgs_dir))
    val_nums = len(os.listdir(val_imgs_dir))
    print("all: " + str(total_nums))
    print("train: " + str(train_nums))
    print("val: " + str(val_nums))


if __name__ == "__main__":
    data_img_dir = "/data/home/zy/huawei/new_data/9_9/images/"
    data_label_dir = "/data/home/zy/huawei/new_data/9_9/labels/"
    #更改文件夹 需要更改顶部的TRAGET_W  TRAGET_H

    train_save_dir = "/data/home/zy/huawei/new_data/" + str(TARGET_W) + "/train_new/"
    val_save_dir = "/data/home/zy/huawei/new_data/" + str(TARGET_W) + "/val_new/"
    test_save_dir = "/data/home/zy/huawei/new_data/" + str(TARGET_W) + "/test_new/"

    train_img_name1 = "382_0_0.png"
    train_img_name2 = "382_0_1.png"
    train_img_name3 = "382_0_2.png"
    train_img_name4 = "382_1_0.png"
    train_img_name5 = "382_1_2.png"
    train_img_name6 = "382_2_0.png"
    train_img_name7 = "382_2_2.png"

    train_img_name1_ = "182_0_0.png"
    train_img_name2_ = "182_0_2.png"
    train_img_name3_ = "182_1_0.png"
    train_img_name4_ = "182_1_2.png"
    train_img_name5_ = "182_2_0.png"
    train_img_name6_ = "182_2_1.png"
    train_img_name7_ = "182_2_2.png"

    val_img_name1 = "382_1_1.png"  # 测试集 三份
    val_img_name2 = "382_2_1.png"  # 测试集 三份
    val_img_name3 = "182_0_1.png"  # 测试集 三份

    test_img_name1 = "182_1_1.png"    # 验证集 1份


    # cut train datas
    cut_images(train_img_name1, os.path.join(data_img_dir, train_img_name1),
               os.path.join(data_label_dir, train_img_name1), train_save_dir)
    cut_images(train_img_name2, os.path.join(data_img_dir, train_img_name2),
               os.path.join(data_label_dir, train_img_name2), train_save_dir)
    cut_images(train_img_name3, os.path.join(data_img_dir, train_img_name3),
               os.path.join(data_label_dir, train_img_name3), train_save_dir)
    cut_images(train_img_name4, os.path.join(data_img_dir, train_img_name4),
               os.path.join(data_label_dir, train_img_name4), train_save_dir)
    cut_images(train_img_name5, os.path.join(data_img_dir, train_img_name5),
               os.path.join(data_label_dir, train_img_name5), train_save_dir)
    cut_images(train_img_name6, os.path.join(data_img_dir, train_img_name6),
               os.path.join(data_label_dir, train_img_name6), train_save_dir)
    cut_images(train_img_name7, os.path.join(data_img_dir, train_img_name7),
               os.path.join(data_label_dir, train_img_name7), train_save_dir)

    cut_images(train_img_name1_, os.path.join(data_img_dir, train_img_name1_),
               os.path.join(data_label_dir, train_img_name1_), train_save_dir)
    cut_images(train_img_name2_, os.path.join(data_img_dir, train_img_name2_),
               os.path.join(data_label_dir, train_img_name2_), train_save_dir)
    cut_images(train_img_name3_, os.path.join(data_img_dir, train_img_name3_),
               os.path.join(data_label_dir, train_img_name3_), train_save_dir)
    cut_images(train_img_name4_, os.path.join(data_img_dir, train_img_name4_),
               os.path.join(data_label_dir, train_img_name4_), train_save_dir)
    cut_images(train_img_name5_, os.path.join(data_img_dir, train_img_name5_),
               os.path.join(data_label_dir, train_img_name5_), train_save_dir)
    cut_images(train_img_name6_, os.path.join(data_img_dir, train_img_name6_),
               os.path.join(data_label_dir, train_img_name6_), train_save_dir)
    cut_images(train_img_name7_, os.path.join(data_img_dir, train_img_name7_),
               os.path.join(data_label_dir, train_img_name7_), train_save_dir)

    cut_images(val_img_name1, os.path.join(data_img_dir, val_img_name1),
               os.path.join(data_label_dir, val_img_name1), val_save_dir)
    cut_images(val_img_name2, os.path.join(data_img_dir, val_img_name2),
               os.path.join(data_label_dir, val_img_name2), val_save_dir)          
    cut_images(val_img_name3, os.path.join(data_img_dir, val_img_name3),
               os.path.join(data_label_dir, val_img_name3), val_save_dir)

    cut_images(test_img_name1, os.path.join(data_img_dir, test_img_name1),
               os.path.join(data_label_dir, test_img_name1), test_save_dir)

    #get_train_val(save_dir)