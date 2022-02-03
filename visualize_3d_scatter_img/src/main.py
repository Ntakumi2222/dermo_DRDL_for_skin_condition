import numpy as np
import os

from src.model.analysisdata.AnalysisData import AnalysisData
from src.model.DL.DLPreProcess import DLPreProcess
from src.data.ImageLoader import ImageLoader
from src.data.ImageWriter import ImageWriter
from src.entity.options.DROption import DROption
from src.manager.DictionaryLearningManager import DictionaryLearningManager
from src.manager.DimensionalityReductionManager import DimensionalityReductionManager
from src.model.QuantitativeEvaluation.QuantitativeEvaluation import QuantitativeEvaluation
from src.model.Remover.RemoveCoefficientAndMappedPoints import RemoveCoefficientAndMappedPoints
from src.visualize.scatterplot.labeler import Labeler3D
from src.visualize.scatterplot.plotly import PlotlyLabeler3D
from src.utils.Utils import append_nparray_except_empty_case, calc_coranking_matrix, cube_coords
from src.settings import UserPref
from src.utils.Utils import save_npz
from src.utils.ArgParser import ArgParser
from src.utils.Utils import save_mnist_for_train_test, load_d_low_d, save_d_c_loss_fig


def main():
    # Read json and set global variables
    arg_parser = ArgParser()
    json_file = arg_parser.get_json_file()
    USER_PREF = UserPref(json_file)

    # MNIST will be downloaded if you use the MNIST DEMO
    if USER_PREF.IS_DEMO:
        save_mnist_for_train_test()

    # main process
    process(USER_PREF)


def process(USER_PREF):
    # 設定値登録
    dr_option = DROption(dr_type=USER_PREF.DR_TYPE, components=3)
    analysis_data = AnalysisData(USER_PREF)
    # 次元削減実行部分
    if USER_PREF.IS_USE_PREPROCESSED_DICTIONARY == False:
        dr_manager = DimensionalityReductionManager(dr_option=dr_option, high_dimentional_data=analysis_data.images, image_paths=analysis_data.image_paths,
                                                    data_original_shape=analysis_data.train_data_original_shape, USER_PREF=USER_PREF)
        analysis_data.mapped_points = dr_manager.get_embeddings()

    if USER_PREF.IS_CALC_CORANKING_MATRIX:
        calc_coranking_matrix(data=analysis_data.images, embeddings=analysis_data.mapped_points,
                              data_type=USER_PREF.DATA_NAME, dr_type=dr_option.dr_type, kappa_s=5, kappa_t=5, time_stamp=USER_PREF.TIME_STAMP)

    # Delete the analysis data corresponding to the index.
    rcmp_instance = RemoveCoefficientAndMappedPoints(analysis_data, USER_PREF)
    analysis_data = rcmp_instance.get_survived_analysis_data()
    # ----------------------------------------
    # The part that performs dictionary learning
    if USER_PREF.IS_USE_PREPROCESSED_DICTIONARY==False:
        # pretraining
        dl_preprocess = DLPreProcess(
            x_train=analysis_data.images, y_train=analysis_data.mapped_points, tau=USER_PREF.TAU, lmd=USER_PREF.LMD, mu=USER_PREF.MU, tol=USER_PREF.TOL, k=USER_PREF.K, epoch=USER_PREF.EPOCHS)
        analysis_data.d_low, analysis_data.d_high = dl_preprocess.get_d_low_d()
        d_loss, c_loss = dl_preprocess.get_d_loss_c_loss()
        save_d_c_loss_fig(d_loss=d_loss, c_loss=c_loss, USER_PREF=USER_PREF)

    if USER_PREF.IS_CHECK_MORPHING:
        analysis_data.add_cube_coordinates()

    reprocess_dl_manager = DictionaryLearningManager(
        x_train=analysis_data.images, x_test=analysis_data.test_images, new_coordinates=analysis_data.new_mapped_points, x_embeddings=analysis_data.mapped_points,
        tau=USER_PREF.TAU,  lmd=USER_PREF.LMD, mu=USER_PREF.MU, K=USER_PREF.K, count=USER_PREF.EPOCHS, d_low=analysis_data.d_low, d_high=analysis_data.d_high)
    analysis_data.images = reprocess_dl_manager.get_images()
    analysis_data.mapped_points = reprocess_dl_manager.get_mapped_points()
    analysis_data.additional_embeddings = reprocess_dl_manager.get_additional_embeddings()

    analysis_data.adapt_image_label_and_paths()
    analysis_data.adjust_centered_image()

    if USER_PREF.IS_ACTIVATE_PLOTLY:
        labeler = PlotlyLabeler3D(
            analysis_data.images, analysis_data.mapped_points, analysis_data.image_labels, analysis_data.image_paths, USER_PREF.DATA_NAME, USER_PREF.DR_TYPE.name, USER_PREF.TIME_STAMP)
        labeler.start()

    if USER_PREF.IS_QUANTITATIVE_EVALUATION:
        QuantitativeEvaluation(mapped_points=analysis_data.mapped_points, image_labels=analysis_data.image_labels,
                               image_paths=analysis_data.image_paths, new_image_labels=analysis_data.new_image_labels, new_reprocess_image_labels=analysis_data.new_reprocess_image_labels, additional_embeddings=analysis_data.additional_embeddings, neighbors=USER_PREF.TAU)

    # 再構成画像の保存
    ImageWriter(images=analysis_data.recons, image_labels=analysis_data.new_image_labels,
                image_paths=analysis_data.new_image_paths, output_dir=os.path.join(USER_PREF.RECON_OUTPUT_DIR, USER_PREF.DATA_NAME))

    save_npz(analysis_data, USER_PREF)


if __name__ == '__main__':
    main()
