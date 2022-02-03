import numpy as np

from src.utils.Utils import append_nparray_except_empty_case


class RemoveCoefficientAndMappedPoints:
    """
    変数公開するときはセットにして出す。（c_lowとembedding, c_highとtrain）
    新しい中間点配列やテスト画像のembeddingが入ってきてから再構成は行われるため、c_highの方が配列数が多い。
    """

    def __init__(self, analysis_data, USER_PREF):
        self.images = analysis_data.images
        self.mapped_points = analysis_data.mapped_points
        self.image_labels = analysis_data.image_labels
        self.image_paths = analysis_data.image_paths
        self.removed_indices = []
        self.survived_analysis_data = analysis_data
        self._remove_delete_indices(USER_PREF.DELETE_INDICES)
        if USER_PREF.IS_USE_PREPROCESSED_DICTIONARY:
            _delete_labels = append_nparray_except_empty_case(
                ["test"], USER_PREF.DELETE_LABELS)
            self._remove_correspond_labels(_delete_labels)
        else:
            self._remove_correspond_labels(USER_PREF.DELETE_LABELS)

    def get_survived_analysis_data(self):
        return self.survived_analysis_data

    def _remove_correspond_labels(self, delete_labels):
        delete_indices = []
        for index, image_label in enumerate(self.image_labels):
            if image_label in delete_labels:
                delete_indices.append(index)
        self._remove(delete_indices)

    def _remove_delete_indices(self, delete_indices):
        self._remove(delete_indices)

    def _remove(self, indices):
        self.removed_indices.extend(indices)
        self.survived_analysis_data.images = np.delete(
            self.images, indices, axis=0)
        self.survived_analysis_data.mapped_points = np.delete(
            self.mapped_points, indices, axis=0)
        self.survived_analysis_data.image_labels = np.delete(
            self.image_labels, indices, axis=0)
        self.survived_analysis_data.image_paths = np.delete(
            self.image_paths, indices, axis=0)
