import numpy as np


class RemoveCoefficientAndMappedPoints:
    """
    変数公開するときはセットにして出す。（c_lowとembedding, c_highとtrain）
    新しい中間点配列やテスト画像のembeddingが入ってきてから再構成は行われるため、c_highの方が配列数が多い。
    """

    def __init__(self, images, mapped_points, image_labels, image_paths):
        self.images = images
        self.mapped_points = mapped_points
        self.image_labels = image_labels
        self.image_paths = image_paths
        self.removed_indices = []
        self.survived_images = []
        self.survived_mapped_points = []
        self.survived_image_labels = []
        self.survived_image_paths = []

    def get_survived_images_mapped_points(self):
        return self.survived_images, self.survived_mapped_points

    def get_survived_image_labels_image_paths(self):
        return self.survived_image_labels, self.survived_image_paths

    def remove_correspond_labels(self, delete_labels):
        delete_indices = []
        for index, image_label in enumerate(self.image_labels):
            if image_label in delete_labels:
                delete_indices.append(index)
        self._remove(delete_indices)

    def remove_delete_indices(self, delete_indices):
        self._remove(delete_indices)

    """
    先にインデックスで削除する方法を取ることとする
    ラベルで削る際は個々の関数を内部的に使う。
    """

    def _remove(self, indices):
        self.removed_indices.extend(indices)
        self.survived_images = np.delete(self.images, indices, axis=0)
        self.survived_mapped_points = np.delete(self.mapped_points, indices, axis=0)
        self.survived_image_labels = np.delete(self.image_labels, indices, axis=0)
        self.survived_image_paths = np.delete(self.image_paths, indices, axis=0)