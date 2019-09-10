import numpy as np

def load_imputation_function(imputation_type):
    print("Imputation type: ", imputation_type)
    if imputation_type == "zero_imp":
        return zero_imputation
    elif imputation_type == "fwd":
        return forward_imputation
    else:
        raise

def zero_imputation(data):
    newdata = []
    for ele in data:
        ele[np.isnan(ele)] = 0
        newdata.append(ele)
    return newdata


def forward_imputation(data):
    for num in range(len(data)):
        mat = data[num].T
        mask = np.isnan(mat)
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        mat[mask] = mat[np.nonzero(mask)[0], idx[mask]]
        data[num] = mat.T
        data[num][np.isnan(data[num])] = 0
    return data
