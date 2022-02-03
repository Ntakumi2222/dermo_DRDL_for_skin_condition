import os

class AnalysisData:
    def __init__(self, USER_PREF) -> None:
        train_data_dir = os.path.join(
            USER_PREF.TRAIN_DATA_DIR, USER_PREF.DATA_NAME)
        
        train_image_loader = ImageLoader(
            train_data_dir, USER_PREF, 'train')
        x_train = train_image_loader.get_image_list()
        # 画像平均と画像の再構成で必要なshape情報
        train_mean_image = train_image_loader.get_mean_image()
        train_data_original_shape = train_image_loader.get_data_original_shape()
        train_file_basenames = train_image_loader.get_image_file_path_list_base_name()

        
        test_data_dir = os.path.join(
            USER_PREF.TEST_DATA_DIR, USER_PREF.DATA_NAME)
        test_image_loader = ImageLoader(test_data_dir, USER_PREF, 'test')
        x_test = test_image_loader.get_image_list()


        # pathの確認で使用するファイル名
        test_file_basenames = test_image_loader.get_image_file_path_list_base_name()
        # 画像のlabel
        image_paths = train_file_basenames
        image_labels = train_image_loader.get_image_labels()
        pass