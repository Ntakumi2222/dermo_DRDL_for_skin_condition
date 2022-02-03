import json
import numpy as np
import pprint

from src.entity.types.DRType import DRType
from src.entity.types.DataPhotoType import DataPhotoType
from src.utils.Utils import Singleton, save_json, get_time_stamp


class UserPref(Singleton):
    """[summary]
    Uneditable class. Singleton class for global variables
    """
    def __init__(self, json_file_path) -> None:
        json_open = open(json_file_path, 'r')
        json_load = json.load(json_open)
        self.NPZ_DATA_DIR = json_load['DIRS']['NPZ_DATA_DIR']
        self.TRAIN_DATA_DIR = json_load['DIRS']['TRAIN_DATA_DIR']
        self.TEST_DATA_DIR = json_load['DIRS']['TEST_DATA_DIR']
        self.JSON_OUTPUT_DIR = json_load['DIRS']['JSON_OUTPUT_DIR']
        self.RECON_OUTPUT_DIR = json_load['DIRS']['RECON_OUTPUT_DIR']
        self.SCORE_OUTPUT_DIR = json_load['DIRS']['SCORE_OUTPUT_DIR']
        self.NPZ_OUTPUT_DIR = json_load['DIRS']['NPZ_OUTPUT_DIR']
        self.LOSS_PLOT_OUTPUT_DIR = json_load['DIRS']['LOSS_PLOT_OUTPUT_DIR']

        self.CSV_DATA_FILE_FOR_QE = json_load['FILES']['CSV_DATA_FILE_FOR_QE']
        self.CSV_DATA_FILE_FOR_MDS = json_load['FILES']['CSV_DATA_FILE_FOR_MDS']
        self.PREPROCESSED_DICTIONARY_PATH = json_load['FILES']['PREPROCESSED_DICTIONARY_PATH']

        self.DATA_NAME = json_load['DR']['DATA_NAME']
        self.DR_TYPE = DRType(json_load['DR']['DR_TYPE'])
        self.DATA_PHOTO_TYPE = DataPhotoType(
            json_load['DR']['DATA_PHOTO_TYPE'])
        self.IS_USE_HSV_VALUE = json_load['DR']['IS_USE_HSV_VALUE']
        self.IS_USE_GRAYSCALE = json_load['DR']['IS_USE_GRAYSCALE']
        self.RELOAD_IMAGES = json_load['DR']['RELOAD_IMAGES']
        self.IMAGE_SCALE = (json_load['DR']['IMAGE_SCALE']['WIDTH'],
                            json_load['DR']['IMAGE_SCALE']['HEIGHT'])

        self.IS_USE_PREPROCESSED_DICTIONARY = json_load['PREFERENCE']['IS_USE_PREPROCESSED_DICTIONARY']
        self.IS_CHECK_MORPHING = json_load['PREFERENCE']['IS_CHECK_MORPHING']
        self.IS_QUANTITATIVE_EVALUATION = json_load['PREFERENCE']['IS_QUANTITATIVE_EVALUATION']
        self.IS_CALC_CORANKING_MATRIX = json_load['PREFERENCE']['IS_CALC_CORANKING_MATRIX']
        self.IS_ACTIVATE_PLOTLY = json_load['PREFERENCE']['IS_ACTIVATE_PLOTLY']
        self.IS_DEMO = json_load['PREFERENCE']['IS_DEMO']
         
        # coordinates_name系は全て一意な値にしてください
        self.REPROCESS_NEW_COORDINATES = np.asarray(
                json_load['DL']['REPROCESS_NEW_COORDINATES'])
        self.K = json_load['DL']['K']
        self.TAU = json_load['DL']['TAU']
        self.MU = json_load['DL']['MU']
        self.LMD = json_load['DL']['LMD']
        self.TOL = json_load['DL']['TOL']
        self.EPOCHS = json_load['DL']['EPOCHS']

        self.DELETE_LABELS = json_load['QE']['DELETE_LABELS']
        self.DELETE_INDICES = json_load['QE']['DELETE_INDICES']
        self.SCORE_ITEMS = json_load['QE']['SCORE_ITEMS']
        self.TIME_STAMP = get_time_stamp()
        # 単一原則からは外れるが、ここでjsonを保存する
        pprint.pprint(json_load)
        save_json(
            f'../result/json/{self.DATA_NAME}_{self.DR_TYPE.name}_{self.TIME_STAMP}.json', json_load)
