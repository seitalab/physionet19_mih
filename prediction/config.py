root = "../data/physionet2019"

scaling = ["no_scale", "standard", "custom"]
imputation = ["mean", "fwd"]
preprocess = [None, "v1", "v2", "v3", "v4", "v5", "v6", "v7"]

dataset_loc = root + "/processed/"
scaler_loc = root + "/scalers/"

result_dir = "./results"
model_dir = "{}/models/".format(result_dir)
save_loc = "{}/predictions/".format(result_dir)
tflog_dir = "{}/runs".format(result_dir)
mlflow_dir = "{}/mlflow".format(result_dir)

ep_str = "ep{:04d}"
modelfile = "{}-{}.pth"
train_logfile = "train_log.txt"
valid_logfile = "valid_log.txt"

validation_raw = root + "/base/test_seed{}.zip"
validation_pred = save_loc + "/predictions.zip"
