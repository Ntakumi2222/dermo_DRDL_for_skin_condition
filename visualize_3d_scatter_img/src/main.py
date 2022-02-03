import numpy as np
import os

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
    
    # Conditional branching through dictionary pre-training
    if USER_PREF.IS_USE_PREPROCESSED_DICTIONARY:
        reprocess(USER_PREF)
    else:
        process(USER_PREF)


"""
TODO:images, mapped_points, image_labels, image_pathsを一つの構造体にする。ImagePackageとか名前つけて
"""


def process(USER_PREF):
    # 設定値登録
    dr_option = DROption(dr_type=USER_PREF.DR_TYPE, components=3)

    train_data_dir = os.path.join(
        USER_PREF.TRAIN_DATA_DIR, USER_PREF.DATA_NAME)
    test_data_dir = os.path.join(USER_PREF.TEST_DATA_DIR, USER_PREF.DATA_NAME)

    # データのロード
    train_image_loader = ImageLoader(
        train_data_dir, USER_PREF, 'train')
    test_image_loader = ImageLoader(test_data_dir, USER_PREF, 'test')

    # 入力画像
    x_train = train_image_loader.get_image_list()
    x_test = test_image_loader.get_image_list()

    # 画像平均と画像の再構成で必要なshape情報
    train_mean_image = train_image_loader.get_mean_image()
    train_data_original_shape = train_image_loader.get_data_original_shape()

    # pathの確認で使用するファイル名
    train_file_basenames = train_image_loader.get_image_file_path_list_base_name()
    test_file_basenames = test_image_loader.get_image_file_path_list_base_name()

    # 画像のlabel
    image_paths = train_file_basenames
    image_labels = train_image_loader.get_image_labels()

    # 次元削減実行部分
    dr_manager = DimensionalityReductionManager(dr_option=dr_option, high_dimentional_data=x_train, image_paths=image_paths,
                                                data_original_shape=train_data_original_shape, USER_PREF=USER_PREF)

    # drの埋め込みベクトルはembeddings, dlの出力ベクトルはmapped_points, 明確な違いがあるため命名を変更している。
    x_embeddings = dr_manager.get_embeddings()

    if USER_PREF.IS_CALC_CORANKING_MATRIX:
        calc_coranking_matrix(data=x_train, embeddings=x_embeddings,
                              data_type=USER_PREF.DATA_NAME, dr_type=dr_option.dr_type, kappa_s=5, kappa_t=5, time_stamp=USER_PREF.TIME_STAMP)

    # 名前をわかりやすいように変更
    images = x_train
    mapped_points = x_embeddings

    # ラベルに対応する画像の削除部分
    rcmp_instance = RemoveCoefficientAndMappedPoints(
        images, mapped_points, image_labels, image_paths)
    rcmp_instance.remove_delete_indices(USER_PREF.DELETE_INDICES)
    rcmp_instance.remove_correspond_labels(USER_PREF.DELETE_LABELS)

    # これまでの変数はsurvivedにほとんど生まれ変わります。
    # 変数上書きしているのはメモリの確保と、前の値は使わないから。必要な場合は適宜保存し使用すること。
    images, mapped_points = rcmp_instance.get_survived_images_mapped_points()
    image_labels, image_paths = rcmp_instance.get_survived_image_labels_image_paths()
    print(f'images:{images.shape}')
    print(f'mapped_points:{mapped_points.shape}')
    print(f'images_labels:{image_labels.shape}')
    print(f'image_paths:{image_paths.shape}')

    # ----------------------------------------
    # ここからが辞書学習再計算
    # 辞書学習再計算実行部分
    # 事前学習
    if USER_PREF.IS_USE_PREPROCESSED_DICTIONARY:
        load_d_low_d(USER_PREF)
    else:
        dl_preprocess = DLPreProcess(
            x_train=images, y_train=mapped_points, tau=USER_PREF.TAU, lmd=USER_PREF.LMD, mu=USER_PREF.MU, tol=USER_PREF.TOL, k=USER_PREF.K, epoch=USER_PREF.EPOCHS)
        d_low, d_high = dl_preprocess.get_d_low_d()
        d_loss, c_loss = dl_preprocess.get_d_loss_c_loss()
        save_d_c_loss_fig(d_loss=d_loss, c_loss=c_loss, USER_PREF=USER_PREF)

    
    # 座標の指定
    new_recoordinates = USER_PREF.REPROCESS_NEW_COORDINATES
    if USER_PREF.IS_CHECK_MORPHING:
        cube_coordinates = cube_coords(x_min=mapped_points.min(axis=0)[0], x_max=mapped_points.max(axis=0)[0],
                                       y_min=mapped_points.min(
                                       axis=0)[1], y_max=mapped_points.max(axis=0)[1],
                                       z_min=mapped_points.min(
                                       axis=0)[2], z_max=mapped_points.max(axis=0)[2],
                                       num=5)
        new_recoordinates = append_nparray_except_empty_case(
            new_recoordinates, cube_coordinates)

    # 新規分の計算
    reprocess_dl_manager = DictionaryLearningManager(
        x_train=images, x_test=x_test, new_coordinates=new_recoordinates, x_embeddings=mapped_points,
        tau=USER_PREF.TAU,  lmd=USER_PREF.LMD, mu=USER_PREF.MU, K=USER_PREF.K, count=USER_PREF.EPOCHS, d_low=d_low, d_high=d_high)
    images = reprocess_dl_manager.get_images()
    mapped_points = reprocess_dl_manager.get_mapped_points()
    additional_embeddings = reprocess_dl_manager.get_additional_embeddings()

    # 画像のlabel追加 train->test->reprocess_new_coord
    new_reprocess_image_labels = np.array(
        ['reprocess' for _ in range(len(new_recoordinates))])
    new_image_labels = append_nparray_except_empty_case(
        test_image_loader.get_image_labels(), new_reprocess_image_labels)
    image_labels = append_nparray_except_empty_case(
        image_labels, new_image_labels)

    new_reprocess_image_paths = np.asarray(
        [f'{new_image_label}_{index}.jpg' for index, new_image_label in enumerate(new_reprocess_image_labels)])
    new_image_paths = append_nparray_except_empty_case(
        test_file_basenames, new_reprocess_image_paths)
    image_paths = append_nparray_except_empty_case(
        image_paths, new_image_paths)

    # 平均画像を足して元の行列に変換
    recons = np.asarray(images[-(len(x_test)+len(new_recoordinates)):])
    images = images + train_mean_image
    recons = recons + train_mean_image
    images = images.reshape(images.shape[0], *train_data_original_shape)
    recons = recons.reshape(recons.shape[0], *train_data_original_shape)

    if USER_PREF.IS_ACTIVATE_PLOTLY:
        labeler = PlotlyLabeler3D(
            images, mapped_points, image_labels, image_paths, USER_PREF.DATA_NAME, USER_PREF.DR_TYPE.name, USER_PREF.TIME_STAMP)
    else:
        labeler = Labeler3D(images, mapped_points, image_labels)
    labeler.start()

    if USER_PREF.IS_QUANTITATIVE_EVALUATION:
        qe = QuantitativeEvaluation(mapped_points=mapped_points, image_labels=image_labels,
                                    image_paths=image_paths, new_image_labels=new_image_labels, new_reprocess_image_labels=new_reprocess_image_labels, additional_embeddings=additional_embeddings, neighbors=USER_PREF.TAU)

    # 再構成画像の保存
    image_writer = ImageWriter(images=recons, image_labels=new_image_labels,
                               image_paths=new_image_paths, output_dir=os.path.join(USER_PREF.RECON_OUTPUT_DIR, USER_PREF.DATA_NAME))
    image_writer.imwrite()

    save_npz(USER_PREF.DATA_NAME, USER_PREF.DR_TYPE, images, train_mean_image, train_data_original_shape,
             mapped_points, image_labels, image_paths, d_high, d_low, USER_PREF)


def reprocess(USER_PREF):
    # 設定値登録
    test_data_dir = os.path.join(USER_PREF.TEST_DATA_DIR, USER_PREF.DATA_NAME)

    # データのロード
    test_image_loader = ImageLoader(test_data_dir, USER_PREF, 'test')

    # 入力画像
    x_test = test_image_loader.get_image_list()

    # pathの確認で使用するファイル名
    test_file_basenames = test_image_loader.get_image_file_path_list_base_name()

    # drの埋め込みベクトルはembeddings, dlの出力ベクトルはmapped_points, 明確な違いがあるため命名を変更している。
    # 名前をわかりやすいように変更
    print('loading npz')
    temp_npz_data = np.load(USER_PREF.PREPROCESSED_DICTIONARY_PATH)

    images = temp_npz_data['images']
    mapped_points = temp_npz_data['mapped_points']
    train_mean_image = temp_npz_data['train_mean_image']
    train_data_original_shape = temp_npz_data['train_data_original_shape']
    image_labels = temp_npz_data['image_labels']
    image_paths = temp_npz_data['image_paths']
    d_low = temp_npz_data['d_low']
    d_high = temp_npz_data['d_high']
    temp_npz_data.close()

    images = images.reshape(images.shape[0], -1)
    images = images - train_mean_image
    rcmp_instance = RemoveCoefficientAndMappedPoints(
        images, mapped_points, image_labels, image_paths)
    # 一度実行するとtestも入ってしまうため、testを削除している。
    _delete_labels = ["test"]
    rcmp_instance.remove_delete_indices(USER_PREF.DELETE_INDICES)
    rcmp_instance.remove_correspond_labels(_delete_labels)

    # これまでの変数はsurvivedにほとんど生まれ変わります。
    # 変数上書きしているのはメモリの確保と、前の値は使わないから。必要な場合は適宜保存し使用すること。
    images, mapped_points = rcmp_instance.get_survived_images_mapped_points()
    image_labels, image_paths = rcmp_instance.get_survived_image_labels_image_paths()

    print(f'images:{images.shape}')
    print(f'mapped_points:{mapped_points.shape}')
    print(f'images_labels:{image_labels.shape}')
    print(f'image_paths:{image_paths.shape}')

    # coranking matrixの計算
    if USER_PREF.IS_CALC_CORANKING_MATRIX:
        calc_coranking_matrix(data=images, embeddings=mapped_points,
                              data_type=USER_PREF.DATA_NAME, dr_type=USER_PREF.DR_TYPE, kappa_s=5, kappa_t=5,time_stamp=USER_PREF.TIME_STAMP)

    # ----------------------------------------
    # ここからが辞書学習再計算
    # 辞書学習再計算実行部分
    # 座標の指定
    new_recoordinates = USER_PREF.REPROCESS_NEW_COORDINATES
    if USER_PREF.IS_CHECK_MORPHING:
        cube_coordinates = cube_coords(x_min=mapped_points.min(axis=0)[0], x_max=mapped_points.max(axis=0)[0],
                                       y_min=mapped_points.min(
                                       axis=0)[1], y_max=mapped_points.max(axis=0)[1],
                                       z_min=mapped_points.min(
                                       axis=0)[2], z_max=mapped_points.max(axis=0)[2],
                                       num=6)
        new_recoordinates = append_nparray_except_empty_case(
            new_recoordinates, cube_coordinates)
    reprocess_dl_manager = DictionaryLearningManager(
        x_train=images, x_test=x_test, new_coordinates=new_recoordinates, x_embeddings=mapped_points,
        tau=USER_PREF.TAU,  lmd=USER_PREF.LMD, mu=USER_PREF.MU, K=USER_PREF.K, count=USER_PREF.EPOCHS, d_low=d_low, d_high=d_high)
    images = reprocess_dl_manager.get_images()
    mapped_points = reprocess_dl_manager.get_mapped_points()
    additional_embeddings = reprocess_dl_manager.get_additional_embeddings()

    # 画像のlabel追加 train->test->reprocess_new_coord
    new_reprocess_image_labels = np.array(
        ['reprocess' for _ in range(len(new_recoordinates))])
    new_image_labels = append_nparray_except_empty_case(
        test_image_loader.get_image_labels(), new_reprocess_image_labels)
    image_labels = append_nparray_except_empty_case(
        image_labels, new_image_labels)

    new_reprocess_image_paths = np.asarray(
        [f'{new_image_label}_{index}.jpg' for index, new_image_label in enumerate(new_reprocess_image_labels)])
    new_image_paths = append_nparray_except_empty_case(
        test_file_basenames, new_reprocess_image_paths)
    image_paths = append_nparray_except_empty_case(
        image_paths, new_image_paths)

    # 平均画像を足して元の行列に変換
    recons = np.asarray(images[-(len(x_test)+len(new_recoordinates)):])
    images = images + train_mean_image
    recons = recons + train_mean_image
    images = images.reshape(images.shape[0], *train_data_original_shape)
    recons = recons.reshape(recons.shape[0], *train_data_original_shape)
    print(recons.shape)
    if USER_PREF.IS_ACTIVATE_PLOTLY:
        labeler = PlotlyLabeler3D(
            images, mapped_points, image_labels, image_paths, USER_PREF.DATA_NAME, USER_PREF.DR_TYPE.name, USER_PREF.TIME_STAMP)
    else:
        labeler = Labeler3D(images, mapped_points, image_labels)
    labeler.start()

    if USER_PREF.IS_QUANTITATIVE_EVALUATION:
        qe = QuantitativeEvaluation(mapped_points=mapped_points, image_labels=image_labels,
                                    image_paths=image_paths, new_image_labels=new_image_labels, new_reprocess_image_labels=new_reprocess_image_labels, additional_embeddings=additional_embeddings, neighbors=5)

    # 再構成画像の保存
    image_writer = ImageWriter(images=recons, image_labels=new_image_labels,
                               image_paths=new_image_paths, output_dir=os.path.join(USER_PREF.RECON_OUTPUT_DIR, USER_PREF.DATA_NAME, USER_PREF.DR_TYPE.name, USER_PREF.TIME_STAMP))
    image_writer.imwrite()
    save_npz(USER_PREF.DATA_NAME, USER_PREF.DR_TYPE, images, train_mean_image, train_data_original_shape,
             mapped_points, image_labels, image_paths, d_high, d_low, USER_PREF)


if __name__ == '__main__':
    main()
