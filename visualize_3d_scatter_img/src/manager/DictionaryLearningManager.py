import numpy as np

from src.model.DL.DLReconstruction import DLReconstruction
from src.model.DL.DLReduction import DLReduction
from src.utils.Utils import append_nparray_except_empty_case
from src.model.DL.DLPreProcess import DLPreProcess


class DictionaryLearningManager:
    """
    注：入力のデータ(x_train, x_embedding, new_coordinates, x_embeddings, d_low, d_high)は全て転置した形になっている。
    """

    def __init__(self, x_train, x_test, new_coordinates, x_embeddings, tau, lmd, mu, K, count, d_low, d_high):
        # 各値の代入
        self.x_train_T = x_train.T
        self.x_test_T = x_test.T
        self.new_coordinates_T = new_coordinates.T
        self.x_embeddings_T = x_embeddings.T
        self.tau = tau
        self.lmd = lmd
        self.mu = mu
        self.K = K
        self.count = count
        self.d_low = d_low
        self.d_high = d_high
        self.additional_embeddings_T = np.asarray([])
        self._execute()

    def get_mapped_points(self):
        return self.mapped_points.T

    def get_images(self):
        return self.images.T

    def get_additional_embeddings(self):
        return self.additional_embeddings_T.T

    def _execute(self):
        # 新規画像の増分的埋め込み
        if self.x_test_T.T.shape[0] != 0:
            self.additional_embeddings_T = self._process_dl_reduction()

        # 追加した画像の座標(増分埋め込み結果, 指定座標)
        self.added_mapped_points = append_nparray_except_empty_case(self.additional_embeddings_T.T,
                                                                    self.new_coordinates_T.T).T
        # 全画像の座標(次元削減結果、(増分埋め込み結果, 指定座標))
        self.mapped_points = append_nparray_except_empty_case(
            self.x_embeddings_T.T, self.added_mapped_points.T).T

        # 再構成
        recons_T = self._process_dl_reconstruction()
         
        self.images = append_nparray_except_empty_case(
            self.x_train_T.T, recons_T.T).T

    def _process_dl_reduction(self):
        print('Process Dictionary Learning Reduction')
        x_test_reduction = self.x_test_T
        dl_reduction = DLReduction(
            self.d_high, self.d_low, x_test_reduction, self.tau, self.lmd, self.mu, self.K)
        new_embeddings_T = dl_reduction.get_y_tests()
        return new_embeddings_T

    def _process_dl_reconstruction(self):
        print('Process Dictionary Learning Reconstruction')
        dl_reconstruction = DLReconstruction(self.d_high, self.d_low, self.added_mapped_points, self.tau, self.lmd,
                                             self.mu, self.K)
        recon_X_T = dl_reconstruction.get_recons()
        return recon_X_T
