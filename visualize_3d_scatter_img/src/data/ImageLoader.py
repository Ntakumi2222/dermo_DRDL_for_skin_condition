import cv2
import glob
import numpy as np
import os
import tqdm

from natsort import natsorted


class ImageLoader:
    def __init__(self, data_dir, USER_PREF, phase):
        self._data_dir = data_dir
        self._USER_PREF = USER_PREF
        self._phase = phase
        self._data_type = USER_PREF.DATA_NAME
        self.data_original_shape = (0, 0, 0)
        self.image_list = []
        self.image_file_path_list = []
        self.image_labels = []
        self.mean_image = []
        if os.path.exists(self._data_dir):
            self._load_images_from_folder()

    def _load_images_from_folder(self):
        print(f'Load {self._phase} images')
        npz_file_path_basename = os.path.join(self._USER_PREF.NPZ_DATA_DIR, self._phase,
                                              self._data_type)
        npz_file_path = npz_file_path_basename + '.npz'
        if not os.path.exists(path=npz_file_path) or self._USER_PREF.RELOAD_IMAGES:
            self.image_file_path_list = natsorted(
                glob.glob(os.path.join(self._data_dir, '**', '*')))
            self.image_labels = [os.path.basename(os.path.dirname(filepath)) for
                                 filepath in self.image_file_path_list]

            for index, filename in enumerate(tqdm.tqdm(self.image_file_path_list)):
                if self._USER_PREF.IS_USE_HSV_VALUE and self._USER_PREF.IS_USE_GRAYSCALE:
                    _img = cv2.resize(cv2.imread(filename, 0), dsize=None, fx=self._USER_PREF.IMAGE_SCALE[0],
                                      fy=self._USER_PREF.IMAGE_SCALE[1])
                    _img = cv2.cvtColor(_img, cv2.COLOR_GRAY2BGR)
                    _img = cv2.cvtColor(_img, cv2.COLOR_RGB2HSV)
                    _img = _img[:, :, 2]
                elif (not self._USER_PREF.IS_USE_HSV_VALUE) and self._USER_PREF.IS_USE_GRAYSCALE:
                    _img = cv2.resize(cv2.imread(filename, 0), dsize=None, fx=self._USER_PREF.IMAGE_SCALE[0],
                                      fy=self._USER_PREF.IMAGE_SCALE[1])
                elif self._USER_PREF.IS_USE_HSV_VALUE and (not self._USER_PREF.IS_USE_GRAYSCALE):
                    _img = cv2.resize(cv2.imread(filename), dsize=None, fx=self._USER_PREF.IMAGE_SCALE[0],
                                      fy=self._USER_PREF.IMAGE_SCALE[1])
                    _img = cv2.cvtColor(_img, cv2.COLOR_BGR2HSV)
                    _img = _img[:, :, 2]
                else:
                    _img = cv2.resize(cv2.imread(filename), dsize=None, fx=self._USER_PREF.IMAGE_SCALE[0],
                                      fy=self._USER_PREF.IMAGE_SCALE[1])

                if index == 0:
                    self.data_original_shape = _img.shape

                _img = _img.flatten()
                if _img is not None:
                    self.image_list.append(_img)

            self.image_list = np.asarray(self.image_list)
            self.mean_image = self.image_list.mean(axis=0)
            self.image_list = self.image_list - self.mean_image
            print(f'Data Shape:{self.image_list.shape}')
            os.makedirs(os.path.dirname(npz_file_path_basename), exist_ok=True)
            np.savez(npz_file_path_basename,
                     image_list=self.image_list,
                     image_file_path_list=self.image_file_path_list,
                     image_labels=self.image_labels,
                     mean_image=self.mean_image,
                     data_original_shape=self.data_original_shape)
        # そのまま使うとdiv errorになることがあるため、一度loadを挟んでいる。
        _temp_npz = np.load(npz_file_path)
        self.image_list = _temp_npz['image_list']
        self.image_file_path_list = _temp_npz['image_file_path_list']
        self.image_labels = _temp_npz['image_labels']
        self.data_original_shape = _temp_npz['data_original_shape']
        self.mean_image = _temp_npz['mean_image']
        pass

    def get_image_file_path_list(self):
        return self.image_file_path_list

    def get_image_file_path_list_base_name(self):
        return np.asarray([os.path.basename(image_file_path) for image_file_path in self.image_file_path_list])

    def get_image_list(self):
        return self.image_list

    def get_data_original_shape(self):
        return self.data_original_shape

    def get_image_labels(self):
        return self.image_labels

    def get_mean_image(self):
        return self.mean_image
