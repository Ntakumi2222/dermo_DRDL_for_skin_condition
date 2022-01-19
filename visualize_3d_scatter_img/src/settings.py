import json
import numpy as np
import pprint

from datetime import datetime
from pytz import timezone
from src.entity.types.DRType import DRType
from src.entity.types.DataPhotoType import DataPhotoType

NPZ_DATA_DIR = ''
TRAIN_DATA_DIR = ''
TEST_DATA_DIR = ''
JSON_OUTPUT_DIR = ''
RECON_OUTPUT_DIR = ''
SCORE_OUTPUT_DIR = ''
NPZ_OUTPUT_DIR = ''
LOSS_PLOT_OUTPUT_DIR = ''
CSV_DATA_FILE_FOR_QE = ''
CSV_DATA_FILE_FOR_MDS = ''
PREPROCESSED_DICTIONARY_PATH = ''

DATA_NAME = ''
DATA_PHOTO_TYPE = None
IS_USE_PREPROCESSED_DICTIONARY = False
IS_CHECK_MORPHING = False
IS_QUANTITATIVE_EVALUATION = True
IS_CALC_CORANKING_MATRIX = True
IS_USE_HSV_VALUE = False
IS_USE_GRAYSCALE = False
RELOAD_IMAGES = True
IS_ACTIVATE_PLOTLY = True


REPROCESS_NEW_COORDINATES = []

IMAGE_SCALE = (1, 1)

K = 0
TAU = 0
MU = 0
LMD = 0
TOL = 0.1
EPOCHS = 0
DR_TYPE = None

DELETE_LABELS = []
DELETE_INDICES = []

SCORE_ITEMS = []
TIME_STAMP = None


def get_NPZ_DATA_DIR(): return NPZ_DATA_DIR
def get_TRAIN_DATA_DIR(): return TRAIN_DATA_DIR
def get_TEST_DATA_DIR(): return TEST_DATA_DIR
def get_JSON_OUTPUT_DIR(): return JSON_OUTPUT_DIR
def get_RECON_OUTPUT_DIR(): return RECON_OUTPUT_DIR
def get_SCORE_OUTPUT_DIR(): return SCORE_OUTPUT_DIR
def get_NPZ_OUTPUT_DIR(): return NPZ_OUTPUT_DIR
def get_LOSS_PLOT_OUTPUT_DIR(): return LOSS_PLOT_OUTPUT_DIR
def get_CSV_DATA_FILE_FOR_QE(): return CSV_DATA_FILE_FOR_QE
def get_CSV_DATA_FILE_FOR_MDS(): return CSV_DATA_FILE_FOR_MDS
def get_PREPROCESSED_DICTIONARY_PATH(): return PREPROCESSED_DICTIONARY_PATH
def get_DATA_NAME(): return DATA_NAME
def get_DATA_PHOTO_TYPE(): return DATA_PHOTO_TYPE
def get_IS_USE_PREPROCESSED_DICTIONARY(): return IS_USE_PREPROCESSED_DICTIONARY
def get_IS_CHECK_MORPHING(): return IS_CHECK_MORPHING
def get_IS_QUANTITATIVE_EVALUATION(): return IS_QUANTITATIVE_EVALUATION
def get_IS_CALC_CORANKING_MATRIX(): return IS_CALC_CORANKING_MATRIX
def get_IS_USE_HSV_VALUE(): return IS_USE_HSV_VALUE
def get_IS_USE_GRAYSCALE(): return IS_USE_GRAYSCALE
def get_RELOAD_IMAGES(): return RELOAD_IMAGES
def get_IS_ACTIVATE_PLOTLY(): return IS_ACTIVATE_PLOTLY
def get_REPROCESS_NEW_COORDINATES(): return REPROCESS_NEW_COORDINATES
def get_IMAGE_SCALE(): return IMAGE_SCALE
def get_K(): return K
def get_TAU(): return TAU
def get_MU(): return MU
def get_LMD(): return LMD
def get_TOL(): return TOL
def get_EPOCHS(): return EPOCHS
def get_DR_TYPE(): return DR_TYPE
def get_DELETE_LABELS(): return DELETE_LABELS
def get_DELETE_INDICES(): return DELETE_INDICES
def get_SCORE_ITEMS(): return SCORE_ITEMS
def get_TIME_STAMP(): return TIME_STAMP

# グローバル変数の書き換えのため修正の際は十分に注意すること


def set_settings_from_json(json_file):
    json_open = open(json_file, 'r')
    json_load = json.load(json_open)
    global NPZ_DATA_DIR
    global TRAIN_DATA_DIR
    global TEST_DATA_DIR
    global JSON_OUTPUT_DIR
    global RECON_OUTPUT_DIR
    global SCORE_OUTPUT_DIR
    global NPZ_OUTPUT_DIR
    global LOSS_PLOT_OUTPUT_DIR
    global CSV_DATA_FILE_FOR_QE
    global CSV_DATA_FILE_FOR_MDS
    global PREPROCESSED_DICTIONARY_PATH
    global DATA_NAME
    global DATA_PHOTO_TYPE
    global IS_USE_PREPROCESSED_DICTIONARY
    global IS_CHECK_MORPHING
    global IS_QUANTITATIVE_EVALUATION
    global IS_CALC_CORANKING_MATRIX
    global IS_USE_HSV_VALUE
    global IS_USE_GRAYSCALE
    global RELOAD_IMAGES
    global IS_ACTIVATE_PLOTLY
    global REPROCESS_NEW_COORDINATES
    global IMAGE_SCALE
    global K
    global TAU
    global MU
    global LMD
    global TOL
    global EPOCHS
    global DR_TYPE
    global DELETE_LABELS
    global DELETE_INDICES
    global SCORE_ITEMS
    global TIME_STAMP

    NPZ_DATA_DIR = json_load['DIRS']['NPZ_DATA_DIR']
    TRAIN_DATA_DIR = json_load['DIRS']['TRAIN_DATA_DIR']
    TEST_DATA_DIR = json_load['DIRS']['TEST_DATA_DIR']
    JSON_OUTPUT_DIR = json_load['DIRS']['JSON_OUTPUT_DIR']
    RECON_OUTPUT_DIR = json_load['DIRS']['RECON_OUTPUT_DIR']
    SCORE_OUTPUT_DIR = json_load['DIRS']['SCORE_OUTPUT_DIR']
    NPZ_OUTPUT_DIR = json_load['DIRS']['NPZ_OUTPUT_DIR']
    LOSS_PLOT_OUTPUT_DIR = json_load['DIRS']['LOSS_PLOT_OUTPUT_DIR']

    CSV_DATA_FILE_FOR_QE = json_load['FILES']['CSV_DATA_FILE_FOR_QE']
    CSV_DATA_FILE_FOR_MDS = json_load['FILES']['CSV_DATA_FILE_FOR_MDS']
    PREPROCESSED_DICTIONARY_PATH = json_load['FILES']['PREPROCESSED_DICTIONARY_PATH']

    DATA_NAME = json_load['DR']['DATA_NAME']
    DR_TYPE = DRType(json_load['DR']['DR_TYPE'])
    DATA_PHOTO_TYPE = DataPhotoType(json_load['DR']['DATA_PHOTO_TYPE'])
    IS_USE_HSV_VALUE = json_load['DR']['IS_USE_HSV_VALUE']
    IS_USE_GRAYSCALE = json_load['DR']['IS_USE_GRAYSCALE']
    RELOAD_IMAGES = json_load['DR']['RELOAD_IMAGES']
    IMAGE_SCALE = (json_load['DR']['IMAGE_SCALE']['WIDTH'],
                   json_load['DR']['IMAGE_SCALE']['HEIGHT'])

    IS_USE_PREPROCESSED_DICTIONARY = json_load['PREFERENCE']['IS_USE_PREPROCESSED_DICTIONARY']
    IS_CHECK_MORPHING = json_load['PREFERENCE']['IS_CHECK_MORPHING']
    IS_QUANTITATIVE_EVALUATION = json_load['PREFERENCE']['IS_QUANTITATIVE_EVALUATION']
    IS_CALC_CORANKING_MATRIX = json_load['PREFERENCE']['IS_CALC_CORANKING_MATRIX']
    IS_ACTIVATE_PLOTLY = json_load['PREFERENCE']['IS_ACTIVATE_PLOTLY']

    # coordinates_name系は全て一意な値にしてください
    REPROCESS_NEW_COORDINATES = np.asarray(
        json_load['DL']['REPROCESS_NEW_COORDINATES'])
    K = json_load['DL']['K']
    TAU = json_load['DL']['TAU']
    MU = json_load['DL']['MU']
    LMD = json_load['DL']['LMD']
    TOL = json_load['DL']['TOL']
    EPOCHS = json_load['DL']['EPOCHS']

    DELETE_LABELS = json_load['QE']['DELETE_LABELS']
    DELETE_INDICES = json_load['QE']['DELETE_INDICES']
    SCORE_ITEMS = json_load['QE']['SCORE_ITEMS']
    TIME_STAMP = get_time_stamp()
    # 単一原則からは外れるが、ここでjsonを保存する
    pprint.pprint(json_load)
    save_json(
        f'../result/json/{DATA_NAME}_{DR_TYPE.name}_{TIME_STAMP}.json', json_load)


def save_json(file_name, json_load):
    with open(file_name, 'w') as f:
        json.dump(json_load, f, indent=4, ensure_ascii=False)


def get_time_stamp():
    utc_now = datetime.now(timezone('UTC'))
    jst_now = utc_now.astimezone(timezone('Asia/Tokyo'))
    ts = jst_now.strftime("%Y%m%d-%H%M%S")
    return ts
