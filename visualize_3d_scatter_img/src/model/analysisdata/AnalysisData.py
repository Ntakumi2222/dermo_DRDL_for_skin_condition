import numpy as np
import os

from src.data.ImageLoader import ImageLoader
from src.utils.Utils import append_nparray_except_empty_case, cube_coords


class AnalysisData:
    def __init__(self, USER_PREF) -> None:
        self.USER_PREF = USER_PREF
        self.images = []
        self.test_images = []
        self.mapped_points = []
        self.train_mean_image = []
        self.train_data_original_shape = (0, 0, 0)
        self.d_low = []
        self.d_high = []

        if USER_PREF.IS_USE_PREPROCESSED_DICTIONARY:
            self._init_for_reprocess()
        else:
            self._init_for_process()
        self._init_for_test()

        self.new_mapped_points = self.USER_PREF.REPROCESS_NEW_COORDINATES
        self.additional_embeddings = []
        self.recons = []

    def _init_for_process(self) -> None:
        train_data_dir = os.path.join(
            self.USER_PREF.TRAIN_DATA_DIR, self.USER_PREF.DATA_NAME)
        self.train_image_loader = ImageLoader(
            train_data_dir, self.USER_PREF, 'train')

        self.images = self.train_image_loader.get_image_list()
        # Shape information needed for image averaging and image reconstruction
        self.train_mean_image = self.train_image_loader.get_mean_image()
        self.train_data_original_shape = self.train_image_loader.get_data_original_shape()
        self.train_file_basenames = self.train_image_loader.get_image_file_path_list_base_name()
        self.image_paths = self.train_file_basenames
        self.image_labels = self.train_image_loader.get_image_labels()

    def _init_for_reprocess(self) -> None:
        _temp_npz_data = np.load(
            self.USER_PREF.PREPROCESSED_DICTIONARY_PATH)
        self.images = _temp_npz_data['images']
        self.mapped_points = _temp_npz_data['mapped_points']
        self.train_mean_image = _temp_npz_data['train_mean_image']
        self.train_data_original_shape = _temp_npz_data['train_data_original_shape']
        self.image_labels = _temp_npz_data['image_labels']
        self.image_paths = _temp_npz_data['image_paths']
        self.d_low = _temp_npz_data['d_low']
        self.d_high = _temp_npz_data['d_high']
        _temp_npz_data.close()

        self.images = self.images.reshape(self.images.shape[0], -1)
        self.images = self.images - self.train_mean_image

    def _init_for_test(self) -> None:
        self.test_data_dir = os.path.join(
            self.USER_PREF.TEST_DATA_DIR, self.USER_PREF.DATA_NAME)
        self.test_image_loader = ImageLoader(
            self.test_data_dir, self.USER_PREF, 'test')

        self.test_images = self.test_image_loader.get_image_list()
        # pathの確認で使用するファイル名
        self.test_file_basenames = self.test_image_loader.get_image_file_path_list_base_name()

    def add_cube_coordinates(self) -> None:
        cube_coordinates = cube_coords(x_min=self.mapped_points.min(axis=0)[0], x_max=self.mapped_points.max(axis=0)[0],
                                       y_min=self.mapped_points.min(
            axis=0)[1], y_max=self.mapped_points.max(axis=0)[1],
            z_min=self.mapped_points.min(
            axis=0)[2], z_max=self.mapped_points.max(axis=0)[2],
            num=5)
        self.new_mapped_points = append_nparray_except_empty_case(
            self.new_mapped_points, cube_coordinates)

    def adapt_image_label_and_paths(self):
        # 画像のlabel追加 train->test->reprocess_new_coord
        _new_reprocess_image_labels = np.array(
            ['reprocess' for _ in range(len(self.new_mapped_points))])
        self.new_image_labels = append_nparray_except_empty_case(
            self.test_image_loader.get_image_labels(), _new_reprocess_image_labels)
        self.image_labels = append_nparray_except_empty_case(
            self.image_labels, self.new_image_labels)

        _new_reprocess_image_paths = np.asarray(
            [f'{new_image_label}_{index}.jpg' for index, new_image_label in enumerate(_new_reprocess_image_labels)])
        self.new_image_paths = append_nparray_except_empty_case(
            self.test_file_basenames, _new_reprocess_image_paths)
        self.image_paths = append_nparray_except_empty_case(
            self.image_paths, self.new_image_paths)

    def adjust_centered_image(self):
        # 平均画像を足して元の行列に変換
        self.recons = np.asarray(
            self.images[-(len(self.test_images)+len(self.new_mapped_points)):])
        self.images = self.images + self.train_mean_image
        self.recons = self.recons + self.train_mean_image
        self.images = self.images.reshape(
            self.images.shape[0], *self.train_data_original_shape)
        self.recons = self.recons.reshape(
            self.recons.shape[0], *self.train_data_original_shape)
