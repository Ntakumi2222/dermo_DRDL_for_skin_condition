import numpy as np
from src.model.DL.DLReconstruction import DLReconstruction
from src.model.DL.DLReduction import DLReduction
from src.utils.Utils import append_nparray_except_empty_case


class ReprocessDictionaryLearningManager:
    def __init__(self, x_train, x_test, new_coordinates, x_embeddings, tau, lmd, mu, image_labels, c_lows, c_highs):
        self.x_train = x_train
        self.x_test = x_test
        self.x_embeddings = x_embeddings
        self.new_coordinates = new_coordinates
        self.tau = tau
        self.lmd = lmd
        self.mu = mu
        self.image_labels = image_labels
        self.c_lows = c_lows
        self.c_highs = c_highs
        self._execute()

    def get_mapped_points(self):
        return self.mapped_points

    def get_images(self):
        return self.images

    def get_additional_embeddings(self):
        return self._additional_embeddings

    def get_added_mapped_points(self):
        return self.added_mapped_points

    def get_recon_x(self):
        return self.recon_x

    def get_c_low_c_high(self):
        return self.c_lows, self.c_highs

    def _execute(self):
        # 新規画像の増分的埋め込み
        self._additional_embeddings = np.asarray([])
        if len(self.x_test) != 0:
            self._additional_embeddings = self._process_dl_reduction()

        # 追加した画像の座標(増分埋め込み結果, 指定座標)
        self.added_mapped_points = append_nparray_except_empty_case(self._additional_embeddings,
                                                                    self.new_coordinates)
        # 全画像の座標(次元削減結果、(増分埋め込み結果, 指定座標))
        self.mapped_points = append_nparray_except_empty_case(self.x_embeddings, self.added_mapped_points)

        # 再構成
        self.recon_x = self._process_dl_reconstruction()
        self.images = append_nparray_except_empty_case(self.x_train, self.recon_x)

    def _process_dl_reduction(self):
        print('ReProcess Dictionary Learning Reduction')
        x_test_reduction = self.x_test
        dl_reduction = DLReduction()
        dl_reduction.process(self.x_train, self.x_embeddings, x_test_reduction, self.tau, self.lmd, self.mu)
        new_embeddings, self.c_lows = dl_reduction.get_new_embed_clows()
        return new_embeddings

    def _process_dl_reconstruction(self):
        print('ReProcess Dictionary Learning Reconstruction')
        dl_reconstruction = DLReconstruction()
        dl_reconstruction.process(self.x_train, self.x_embeddings, self.added_mapped_points, self.tau, self.lmd,
                                  self.mu)
        recon_X, self.c_highs = dl_reconstruction.get_recons()
        return recon_X
