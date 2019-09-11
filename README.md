# Location of challenge data

Place unzipped challenge data at 
`./data/physionet2019/raw`

# Data preparation

1. Run `01_makedataset.py`
2. Run `02_convert_data.py`
      Choose preprocess type, imputation type and scaler type.
<<<<<<< HEAD
      
=======
      - <scaling> : Scaling function to call from `custom_scalers.py`
      
            - 0: no scaling
            - 1: standard scaler (default)
            - 2: custom
      - <imputation> : Imputation function to call from `imputation_functions.py`
      
            - 0: mean imputation (default)
            - 1: forward imputation
      - <preprocess> : Preprocess function to call from `manual_preprocessor.py`
      
            - 0: No preprocess
            - 4: Dafault
            - For details, please check `manual_preprocessor.py`
            
      - <seed> : Number of random seed to split data (Default: 1)

# 4. model training
1. `cd prediction/`
2. `python train.py`
      (By default hyper parameter settings, model achieves following normalized utility score.)
      ```
      train,valid,test
      0.431469,0.415992,0.420764
      ```
>>>>>>> 1636956... Update README.md
