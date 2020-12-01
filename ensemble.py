import argparse
from multiprocessing.pool import Pool

import numpy as np
from cv2 import cv2

cv2.setNumThreads(0)
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def average_strategy(images):
    return np.average(images, axis=0)


def hard_voting(images):
    rounded = np.round(images / 255.)
    return np.round(np.sum(rounded, axis=0) / images.shape[0]) * 255.


def add_weight(images):
    images = np.sum(images, axis=0)
    images[images <= 255.] = 0
    images[images > 255.] = 1

    return images*255.


def ensemble_image(params):
    file, dirs, ensemble_dir, strategy = params
    temp = 'huawei/test_result'
    images = []
    for dir in dirs:
        file_path = os.path.join(temp, dir, file)
        images.append(cv2.imread(file_path, cv2.IMREAD_COLOR))
    images = np.array(images)

    if strategy == 'average':
        ensembled = average_strategy(images)
    elif strategy == 'hard_voting':
        ensembled = hard_voting(images)
    elif strategy == 'add_weight':
        ensembled = add_weight(images)

    cv2.imwrite(os.path.join(ensemble_dir, file), ensembled)


def ensemble(dirs, strategy, ensemble_dir, n_threads):
    result_dir = 'huawei/test_result/1024test_result'
    files = [
        file for file in os.listdir(result_dir) if file.split('_')[0] == 'test'
    ]

    params = []
    for file in files:
        params.append((file, dirs, ensemble_dir, strategy))
    pool = Pool(n_threads)
    pool.map(ensemble_image, params)


test_dirs = ['256test_result', '1024test_result_2']

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Ensemble masks")
    arg = parser.add_argument
    arg('--ensembling_cpu_threads', type=int, default=8)
    arg('--ensembling_dir',
        type=str,
        default='huawei/ensemble_res34v5_dink34fpn')
    arg('--strategy', type=str, default='add_weight')
    arg('--dirs_to_ensemble', nargs='+', default=test_dirs)
    args = parser.parse_args()

    dirs = args.dirs_to_ensemble
    if not os.path.exists(args.ensembling_dir):
        os.makedirs(args.ensembling_dir)

    ensemble(dirs, args.strategy, args.ensembling_dir,
             args.ensembling_cpu_threads)
