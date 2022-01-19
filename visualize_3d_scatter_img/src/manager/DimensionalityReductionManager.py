

from src.settings import get_TIME_STAMP
from src.model.DR.DRModel import DRModel
from src.model.QuantitativeEvaluation.CorankingMatrix import CoRankingMatrix


class DimensionalityReductionManager:
    def __init__(self, dr_option, data, image_paths, data_original_shape, data_type):
        self.data = data
        self.dr_option = dr_option
        self._data_type = data_type
        print('Process Dimensionary Reduction')
        self.dr_model = DRModel(dr_option=dr_option, data=data, image_paths=image_paths, data_original_shape=data_original_shape,
                                data_type=self._data_type)
        self.embeddings = self.dr_model.get_embeddings()
        print('Done')

    def get_embeddings(self):
        return self.embeddings

