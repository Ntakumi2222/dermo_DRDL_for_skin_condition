import cv2
import io
import json
import numpy as np
import os
import PIL.Image
import seaborn as sns
import time
import tqdm

from datetime import datetime
from matplotlib import pyplot as plt
from pytz import timezone

from src.model.QuantitativeEvaluation.CorankingMatrix import CoRankingMatrix

"""
class part
"""


class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.end_time = time.time()

    def stop(self):
        self.end_time = time.time()
        self.display_erapsed_time()

    def restart(self):
        self.start_time = time.time()
        self.end_time = time.time()

    def display_erapsed_time(self):
        eraped_time = self.end_time - self.start_time
        if eraped_time == 0:
            print('Timer is moving!')
        else:
            print(f'Erapsed Time:{eraped_time}')


class Singleton(object):
    def __new__(cls, *args, **kargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance


"""
function part
"""


def append_nparray_except_empty_case(*args):
    ol = []
    for l in args:
        if len(l) > 0:
            ol.append(l)
    if len(ol) > 0:
        return np.concatenate(ol)
    else:
        return np.array([])


def args_to_nparray(*args):
    nparray = np.asarray([item for item in args])
    return nparray


def calc_intermediate_coord(coord1, coord2, iter_num):
    dist_coords = [coord1]
    coord1 = np.asarray(coord1)
    coord2 = np.asarray(coord2)
    length = (coord1 - coord2) / iter_num
    for i in range(iter_num):
        dist_coords.append(coord1 - length * i)
    dist_coords.append(coord2)
    return np.asarray(dist_coords)


def calc_coranking_matrix(data, embeddings, data_type, dr_type, kappa_s, kappa_t, time_stamp):
    cr = CoRankingMatrix(data)
    cr_value = cr.evaluate_corank_matrix(embeddings, kappa_s, kappa_t)
    print(cr_value)
    heatmap_coranking_matrix_df = cr.multi_evaluate_corank_matrix(
        embeddings, range(2, 96), range(2, 88))

    plt.figure()
    plt.title(f's:{kappa_s}, t:{kappa_t}, value:{cr_value}')
    sns.heatmap(heatmap_coranking_matrix_df)
    plt.savefig(
        f'../result/corank/{data_type}_{dr_type.name}_{time_stamp}.png')
    plt.close('all')
    plt.show()


def cube_coords(x_min, x_max, y_min, y_max, z_min, z_max, num):
    coords = []
    for x in np.linspace(x_min, x_max, num):
        for y in np.linspace(y_min, y_max, num):
            for z in np.linspace(z_min, z_max, num):
                coords.append([x, y, z])
    coords = np.asarray(coords)
    return coords


def compress_to_bytes(data, fmt):
    """
    Helper function to compress image data via PIL/Pillow.
    """
    buff = io.BytesIO()
    scale = 255.0 / np.max(data)
    img = PIL.Image.fromarray(np.uint8(data * scale))
    img.save(buff, format=fmt)
    return buff.getvalue()


def calculate_image_grayscale(img):
    dst = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype('uint8')
    return dst


def convert_bgr_to_grayscale(img):
    gray = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    return gray


def convert_grayscale_to_heatmap(img):
    return cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_JET)


def get_time_stamp():
    utc_now = datetime.now(timezone('UTC'))
    jst_now = utc_now.astimezone(timezone('Asia/Tokyo'))
    ts = jst_now.strftime("%Y%m%d-%H%M%S")
    return ts


def save_npz(analysis_data, USER_PREF):
    os.makedirs(USER_PREF.NPZ_OUTPUT_DIR, exist_ok=True)
    np.savez(os.path.join(USER_PREF.NPZ_OUTPUT_DIR,
             f'{USER_PREF.DATA_NAME}_{USER_PREF.DR_TYPE.name}_{USER_PREF.TIME_STAMP}'),
             images=analysis_data.images, train_mean_image=analysis_data.train_mean_image, train_data_original_shape=analysis_data.train_data_original_shape, mapped_points=analysis_data.mapped_points, image_labels=analysis_data.image_labels, image_paths=analysis_data.image_paths, d_high=analysis_data.d_high, d_low=analysis_data.d_low)
    pass


def save_json(file_name, json_load):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as f:
        json.dump(json_load, f, indent=4, ensure_ascii=False)


def save_mnist_for_train_test():
    from torchvision import datasets
    # Destination folder settings
    rootdir = "../data"
    traindir = os.path.join(rootdir, "train_data", "MNIST")
    testdir = os.path.join(rootdir, "test_data", "MNIST")

    print("MNIST dataset loading")
    train_dataset = datasets.MNIST(root=rootdir, train=True, download=True)
    test_dataset = datasets.MNIST(root=rootdir, train=False, download=True)

    print("Save image as train")
    dict_mnist_count = {0: 0, 1: 0, 2: 0, 3: 0,
                        4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    train_tol = 100
    for img, label in train_dataset:
        if dict_mnist_count[label] < train_tol:
            dict_mnist_count[label] += 1
            savedir = os.path.join(traindir , str(label))
            os.makedirs(savedir, exist_ok=True)
            savepath = os.path.join(savedir, str(
                dict_mnist_count[label]).zfill(5) + ".png")
            img.save(savepath)
        else:
            continue

    print("Save image as test")
    dict_mnist_count = {0: 0, 1: 0, 2: 0, 3: 0,
                        4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    test_tol = 10
    for img, label in test_dataset:
        if dict_mnist_count[label] < test_tol:
            dict_mnist_count[label] += 1
            savedir = os.path.join(testdir, f'{str(label)}_test')
            os.makedirs(savedir, exist_ok=True)
            savepath = os.path.join(savedir,
                                    str(dict_mnist_count[label]).zfill(5) + ".png")
            img.save(savepath)
        else:
            continue


def save_d_c_loss_fig(d_loss, c_loss, USER_PREF):
    plt.plot(range(len(d_loss)), d_loss,
             marker="o", color="red", linestyle="--", label='D loss')
    plt.plot(range(len(c_loss)), c_loss,
             marker="v", color="blue", linestyle=":", label='C loss')
    plt.title('D loss and C loss per epoch')
    plt.xlabel("epoch")
    plt.ylabel("loss(log)")
    plt.yscale('log')
    plt.legend()
    os.makedirs(USER_PREF.LOSS_PLOT_OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(USER_PREF.LOSS_PLOT_OUTPUT_DIR,
                f'{USER_PREF.DATA_NAME}_{USER_PREF.DR_TYPE.name}_{USER_PREF.TIME_STAMP}.png'), format="png", dpi=300)
    plt.cla()


def load_d_low_d(USER_PREF):
    print('loading npz')
    _temp_npz_data = np.load(USER_PREF.PREPROCESSED_DICTIONARY_PATH)
    d_low = _temp_npz_data['d_low']
    d_high = _temp_npz_data['d_high']
    _temp_npz_data.close()
    return d_low, d_high
