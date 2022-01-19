import pandas as pd
import umap
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.manifold import Isomap, TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from src.entity.types.DRType import DRType
from src.entity.options.DROption import DROption
from src.settings import get_DATA_PHOTO_TYPE, get_IS_USE_HSV_VALUE, get_IS_USE_GRAYSCALE, get_CSV_DATA_FILE_FOR_MDS, get_SCORE_ITEMS, get_TIME_STAMP
from src.utils.Utils import convert_bgr_to_grayscale, convert_grayscale_to_heatmap, calculate_image_grayscale
from src.model.DR.MethodFromScratch.mds import mds
from src.model.DR.MethodFromScratch.isomap import isomap


class DRModel:
    def __init__(self, dr_option: DROption, data: list, image_paths: list, data_original_shape=(3000, 4500), data_type: str = 'rgb'):
        self._dr_option = dr_option
        self._data = data
        self._data_original_shape = data_original_shape
        self._data_type = data_type
        self._embeddings = []
        self.image_paths = image_paths
        self._process()

    def _process(self):
        if self._dr_option.dr_type == DRType.ISOMAP:
            self._isomap()
        elif self._dr_option.dr_type == DRType.TSNE:
            self._tsne()
        elif self._dr_option.dr_type == DRType.PCA:
            self._pca()
        elif self._dr_option.dr_type == DRType.UMAP:
            self._umap()
        elif self._dr_option.dr_type == DRType.MDS:
            self._mds()
    """
    def _isomap(self):
        self._embeddings, matrix_w, cr = isomap(data=self._data,
                                                n_components=self._dr_option.argments.components, n_neighbors=self._dr_option.argments.n_neighbors,)
        self.save_latent(
            contribution_rate=cr, output_path=f'../result/isomap/latent/{self._data_type}_latent_{get_TIME_STAMP()}.png')

    """
    def _isomap(self):
        isomap_instance = Isomap(
            n_components=self._dr_option.argments.components, n_neighbors=self._dr_option.argments.n_neighbors,
            neighbors_algorithm=self._dr_option.argments.neighbors_algorithm)
        self._embeddings = isomap_instance.fit_transform(self._data)
        print(isomap_instance.reconstruction_error())
    
    def _tsne(self):
        tsne_instance = TSNE(n_components=self._dr_option.argments.components,
                             perplexity=self._dr_option.argments.perplexity, n_iter=self._dr_option.argments.n_iter,
                             learning_rate=self._dr_option.argments.learning_rate, init=self._dr_option.argments.init)
        self._embeddings = tsne_instance.fit_transform(self._data)

    def _umap(self):
        umap_instance = umap.UMAP(n_components=self._dr_option.argments.components,
                                  n_neighbors=self._dr_option.argments.n_neighbors,
                                  min_dist=self._dr_option.argments.min_dist)
        self._embeddings = umap_instance.fit_transform(self._data)

    def _pca(self):
        pca_instance = PCA(n_components=self._dr_option.argments.components, whiten=self._dr_option.argments.whiten,
                           svd_solver='full')
        self._embeddings = pca_instance.fit_transform(self._data)
        # 寄与率計算の部分
        pca = PCA(n_components=50, whiten=False, svd_solver='full')
        pca.fit_transform(self._data)
        cr = pca.explained_variance_ratio_
        self.save_latent(contribution_rate=cr,
                          output_path=f'../result/pca/latent/{self._data_type}_latent_{get_TIME_STAMP()}.png')

        fig = plt.figure(figsize=(100, 60), dpi=200)
        row_length = 5
        col_length = 10
        for i in range(row_length):
            for j in range(col_length):
                plt.subplot(row_length, col_length, 1 + col_length * i + j)
                if get_IS_USE_HSV_VALUE() or get_IS_USE_GRAYSCALE():
                    gray = pca.components_[1 + row_length * i + j, :].reshape(
                        self._data_original_shape)
                else:
                    gray = convert_bgr_to_grayscale(pca.components_[1 + row_length * i + j, :].reshape(
                        self._data_original_shape))
                gray_scaled = calculate_image_grayscale(gray)
                heatmap = convert_grayscale_to_heatmap(gray_scaled)
                plt.imshow(heatmap)
                plt.axis("off")
        plt.savefig(
            f'../result/pca/coeff/{self._data_type}_coeff_heatmap_{get_TIME_STAMP()}.png', dpi=200)
        plt.close('all')

    def _mds(self):
        score_df = pd.read_excel(
            get_CSV_DATA_FILE_FOR_MDS(), sheet_name=1, index_col=0, engine='openpyxl')
        score_list = []
        for image_path in self.image_paths:
            temp_score = score_df[score_df[f'実ファイル名({get_DATA_PHOTO_TYPE().value})']
                                  == image_path][get_SCORE_ITEMS()].values[0]
            score_list.append(temp_score)
        score_list = np.asarray(score_list)
        mds_instance = MDS(
            n_components=self._dr_option.argments.components,
            metric=self._dr_option.argments.metric,
            dissimilarity=self._dr_option.argments.dissimilarity
        )
        self._embeddings = mds_instance.fit_transform(score_list)
    """
    def _mds(self):
        score_df = pd.read_excel(
            get_CSV_DATA_FILE_FOR_MDS(), sheet_name=1, index_col=0, engine='openpyxl')
        score_list = []
        for image_path in self.image_paths:
            temp_score = score_df[score_df[f'実ファイル名({get_DATA_PHOTO_TYPE().value})']
                                  == image_path][get_SCORE_ITEMS()].values[0]
            score_list.append(temp_score)
        score_list = np.asarray(score_list)
        score_list = squareform(pdist(score_list))
        self._embeddings, matrix_w, cr = mds(data=score_list,
                                             n_components=self._dr_option.argments.components,
                                             )
        self.save_latent(
            contribution_rate=cr, output_path=f'../result/mds/latent/{self._data_type}_latent_{get_TIME_STAMP()}.png')
    """
    def save_latent(self, contribution_rate, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print('各次元の寄与率: {0}'.format(contribution_rate))
        print('累積寄与率: {0}'.format(
            sum(contribution_rate[0:self._dr_option.argments.components])))
        fig = plt.figure()
        plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
        plt.plot([0] + list(np.cumsum(contribution_rate)), "-o")
        plt.xlabel("Number of principal components")
        plt.ylabel("Cumulative contribution rate")
        plt.title(
            f'contribution rate up to three dimensions: {sum(contribution_rate[0:self._dr_option.argments.components])}')
        plt.grid()
        plt.savefig(output_path)
        pass

    def get_embeddings(self):
        return self._embeddings
