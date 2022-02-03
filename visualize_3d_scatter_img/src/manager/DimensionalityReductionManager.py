

from src.model.DR.DRModel import DRModel
from src.model.QuantitativeEvaluation.CorankingMatrix import CoRankingMatrix


class DimensionalityReductionManager:
    def __init__(self, dr_option, high_dimentional_data, image_paths, data_original_shape, USER_PREF):
        self.data = high_dimentional_data
        self.dr_option = dr_option
        self.USER_PREF = USER_PREF
        self._data_type = self.USER_PREF.DATA_NAME
        print('Process Dimensionary Reduction')
        self.dr_model = DRModel(dr_option=dr_option, data=high_dimentional_data, image_paths=image_paths, data_original_shape=data_original_shape,
                                USER_PREF=self.USER_PREF)
        self.embeddings = self.dr_model.get_embeddings()
        print('Done')

    def get_embeddings(self):
        return self.embeddings

