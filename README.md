# 1. Environment settings

Run `pip install -r requirements.txt`

# 2. Place challenge data

1. Place challenge data at `./data/physionet2019/raw/`.
2. Unzip challenge data.

# 3. Data preparation

1. `cd data_prep/`
2. Run `python 01_makedataset.py`
3. Run `python 02_convert_data.py <scaling> <imputation> <preprocess> <seed>`

      Choose preprocess type, imputation type and scaler type.
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
