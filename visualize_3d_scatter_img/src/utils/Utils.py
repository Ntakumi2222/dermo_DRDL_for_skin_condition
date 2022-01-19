import cv2
import io
import numpy as np
import os
import PIL.Image
import time
import seaborn as sns

from matplotlib import pyplot as plt
from src.model.QuantitativeEvaluation.CorankingMatrix import CoRankingMatrix
from src.settings import get_NPZ_OUTPUT_DIR, get_TIME_STAMP

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


def save_npz(DATA_TYPE, DR_TYPE, images, train_mean_image, train_data_original_shape, mapped_points, image_labels, image_paths, d_high, d_low):
    os.makedirs(get_NPZ_OUTPUT_DIR(), exist_ok=True)
    np.savez(os.path.join(get_NPZ_OUTPUT_DIR(),
             f'{DATA_TYPE}_{DR_TYPE.name}_{get_TIME_STAMP()}'),
             images=images, train_mean_image=train_mean_image, train_data_original_shape=train_data_original_shape, mapped_points=mapped_points, image_labels=image_labels, image_paths=image_paths, d_high=d_high, d_low=d_low)
    pass


def calc_coranking_matrix(data, embeddings, data_type, dr_type, kappa_s, kappa_t):
    cr = CoRankingMatrix(data)
    cr_value = cr.evaluate_corank_matrix(embeddings, kappa_s, kappa_t)
    print(cr_value)
    heatmap_coranking_matrix_df = cr.multi_evaluate_corank_matrix(
        embeddings, range(2, 96), range(2, 88))

    plt.figure()
    plt.title(f's:{kappa_s}, t:{kappa_t}, value:{cr_value}')
    sns.heatmap(heatmap_coranking_matrix_df)
    plt.savefig(
        f'../result/corank/{data_type}_{dr_type.name}_{get_TIME_STAMP()}.png')
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
