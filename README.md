
# DRDL
A tool for applying dimensionality reduction to dictionary learning.

# Original Paper
A Dictionary-Based Algorithm for Dimensionality Reduction and Data Reconstruction(https://ieeexplore.ieee.org/abstract/document/6976986)

# Features
DRDL allows incremental dimensionality reduction.

# Requirement
* certifi==2021.10.8
* python = 3.8
* matplotlib==3.5.1
* natsort==8.1.0
* numpy==1.22.1
* opencv-contrib-python==4.5.5.62
* pandas==1.4.0
* Pillow==9.0.0
* plotly==5.5.0
* pytz==2021.3
* scikit-learn==1.0.2
* seaborn==0.11.2
* torch==1.10.2
* torchvision==0.11.3
* tqdm==4.62.3
* umap-learn==0.5.2


# Installation
Create an environment with anaconda, pyenv, or docker, and then type the following command
```bash
pip install -r requirements.txt
```

# Usage
Prepare settings.json and the corresponding folder before running
```bash
python -m src.main --json ../data/json/settings.json
```
The configuration of settings.json is as follows
```json
{
    # Specifies the directory to be used for input/output.
    "DIRS":{
        "NPZ_DATA_DIR" : "../data/npz",
        "TRAIN_DATA_DIR" : "../data/train_data",
        "TEST_DATA_DIR" : "../data/test_data",
        "JSON_OUTPUT_DIR" : "../result/json",
        "RECON_OUTPUT_DIR" : "../result/recon_img",
        "SCORE_OUTPUT_DIR" : "../result/score",
        "NPZ_OUTPUT_DIR" : "../result/npz/recon_data",
        "LOSS_PLOT_OUTPUT_DIR" : "../result/loss_plot"
    },

    # Specifies the file to be used for input/output.
    ※As for the score, it is the author's score.
    Please modify drmodel.py and QuantitativeEvaluation.py 
    accordingly to support xlsx.
    "FILES":{
        "CSV_DATA_FILE_FOR_QE" : "../data/csv/test.xlsx",
        "CSV_DATA_FILE_FOR_MDS" : "../data/csv/test.xlsx",
        "PREPROCESSED_DICTIONARY_PATH" : "../result/npz/recon_data/MNIST_UMAP_20220203-134902.npz"
    },

    # Parameters used for dimensionality reduction and constants used for storage.
    "DR":{
        # Folder name used by train and test
        "DATA_NAME" : "MNIST",
        # Type of Dimentionality Reduction(PCA, ISOMAP, t-SNE, UMAP) MDS is for scores only.
        "DR_TYPE" : "UMAP",
        # Dermo-Image Type(rgb, pol, uv, concate)
        "DATA_PHOTO_TYPE" : "rgb",
        # true if you want to convert HSV image
        "IS_USE_HSV_VALUE" : false,
        # true if you want to convert Grayscale image
        "IS_USE_GRAYSCALE" : false,
        # false if you want to use the saved npy data
        "RELOAD_IMAGES" : true,
        # When using dermo-images, it is recommended to use 0.2~0.3 in terms of computational complexity.
        "IMAGE_SCALE" : {
            "WIDTH" : 1.0, 
            "HEIGHT": 1.0
        }
    },

    # The type of processing to be done can be specified by a boolean value.
    "PREFERENCE":{
        # Compute using existing dictionary training results
        "IS_USE_PREPROCESSED_DICTIONARY" : false,
        # This allows morphing plots to be evenly distributed for reconstruction
        "IS_CHECK_MORPHING" : false,
        # Evaluate a new image according to its score.（If the score is not provided, an error will occur.）
        "IS_QUANTITATIVE_EVALUATION" : false,
        # Calculates the CoRanking Matrix
        "IS_CALC_CORANKING_MATRIX" : true,
        # Visualize the plot with plotly.
        "IS_ACTIVATE_PLOTLY" : true,
        # Download MNIST for DEMO
        "IS_DEMO": true
    },

   # Parameters used for dictionary learning 
    "DL":{
        # Arbitrary coordinates can be reconstructed. (Please specify in a two-dimensional array. e.g:[[0,0,0],[1,1,1]])
        "REPROCESS_NEW_COORDINATES" : [],
        # Capacity to be provided by the dictionary
        "K" : 100,
        # Number of images to be referenced in one epoch
        "TAU" : 10,
        # The degree to which C is considered, increases the stability of the calculation.
        "MU" : 0.05,
        # Degree of inclusion in updating low-dimensional space and high-dimensional space
        "LMD" : 0.5,
        # Tolerance for early stopping
        "TOL": 0.1,
        # Epochs
        "EPOCHS" : 100
    },
    "QE":{
        # Labels to be erased. e.g['0','1']
        "DELETE_LABELS" : [],
        # Indices to be erased. e.g[0,1,2,3]
        "DELETE_INDICES" : [],
        # Used to remove labels related to a score specified by you.
        "SCORE_ITEMS" : []
    }
}
```
# Note
The score is for the dataset used in the authors' paper and should be changed accordingly.

# Author
* Takumi Nishizawa
* National Institute of Informatics
* t-nishizawa-n2t@nii.ac.jp

# License
"DRDL" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).

# Reference
* A Dictionary-Based Algorithm for Dimensionality Reduction and Data Reconstruction(https://ieeexplore.ieee.org/abstract/document/6976986)
* How to Evaluate Dimensionality Reduction? - Improving the Co-ranking Matrix(https://arxiv.org/abs/1110.3917)