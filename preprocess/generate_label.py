import pandas as pd
import numpy as np
from PIL import Image
import os
import cv2 as cv


def has_label(img):
    img = np.array(img)
    if np.min(img) == 0:
        return True
    else:
        return False


def func(label_path):
    result_list = []
    label_path_list = [f for f in os.listdir(label_path)]
    print(len(label_path_list))
    for f in label_path_list:
        temp_dict = {}
        data = Image.open(os.path.join(label_path, f))
        temp_dict['fname'] = f
        if has_label(data):
            temp_dict['street'] = 1
        else:
            temp_dict['street'] = 0
        print(temp_dict)
        result_list.append(temp_dict)

    df = pd.DataFrame(result_list, columns=['fname', 'street'])
    df.to_csv('./val_label.csv', index=False)


def read_from_csv(filepath, id):
    file_df = pd.read_csv(filepath)

    imgfile = file_df['fname'][id]
    print('{}, label:{}'.format(imgfile, file_df['street'][id]))
    image_path = '/home/xjw/codingFiles/Python/HuaweiYun_Competetion/DlinkNet/huawei/val/images'
    label_path = '/home/xjw/codingFiles/Python/HuaweiYun_Competetion/DlinkNet/huawei/val/labels'

    label = os.path.join(label_path, imgfile)
    data_path = os.path.join(image_path, imgfile)
    img = cv.imread(data_path)
    mask = cv.imread(label)
    cv.imshow('img', img)
    cv.imshow('mask', mask*255)
    cv.waitKey()
    cv.destoryAllWindows(0)

if __name__ == "__main__":
    label_path = '/home/xjw/codingFiles/Python/HuaweiYun_Competetion/DlinkNet/huawei/val/labels'
    image_path = '/home/xjw/codingFiles/Python/HuaweiYun_Competetion/DlinkNet/huawei/val/images'
    # func(label_path)
    read_from_csv('./val_label.csv',62)