import os, pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from glob import glob
from custom_scalers import load_scaler_function
from imputation_functions import load_imputation_function
from manual_preprocessor import load_preprocess_function
from prep_umatrix import UtilityMatrixMaker


class DataConverter(object):

    dirname = "physionet2019"
    procloc = "processed"
    scaler_dir = "scalers"

    def __init__(self, root, scaling, imputation, preprocess=None, seed=1):
        self.seed = seed
        self.root = root

        self.base_loc = os.path.join(self.root, self.dirname, self.procloc)
        processtype = self.form_processtype(scaling, imputation, preprocess)
        
        self.data_loc = os.path.join(self.base_loc, "base")
        self.save_loc = os.path.join(self.base_loc, processtype)
        self.scaler_loc = os.path.join(self.root, self.dirname, 
                                       self.scaler_dir, processtype)
        os.makedirs(self.save_loc, exist_ok=True)
        os.makedirs(self.scaler_loc, exist_ok=True)

        self.umat_maker = UtilityMatrixMaker()
        self.scaling = scaling
        self.imputation = imputation
        self.preprocess = preprocess
        self.imputation_function = load_imputation_function(self.imputation)
        self.preprocess_function = load_preprocess_function(self.preprocess)
        
    def form_processtype(self, scaling, imputation, preprocess):
        processtype = "{}-{}".format(imputation, scaling)
        if preprocess is not None:
            processtype += "-{}".format(preprocess)
        return processtype
    
    def load_pickle(self, datatype):
        loadinfo = (datatype, self.seed)
        Xfile = "/X_{}_seed{}.pkl".format(*loadinfo)
        X = pickle.load(open(self.data_loc + Xfile, "rb"))
        yfile = "/y_{}_seed{}.pkl".format(*loadinfo)
        y = pickle.load(open(self.data_loc + yfile, "rb"))
        Ifile = "/I_{}_seed{}.pkl".format(*loadinfo)
        I = pickle.load(open(self.data_loc + Ifile, "rb"))
        return X, y, I

    def save_pickle(self, X, y, M, U, I, datatype):
        saveinfo = (datatype, self.seed)
        Xfile = "/X_{}_seed{}.pkl".format(*saveinfo)
        pickle.dump(X, open(self.save_loc + Xfile, "wb"))
        yfile = "/y_{}_seed{}.pkl".format(*saveinfo)
        pickle.dump(y, open(self.save_loc + yfile, "wb"))
        Mfile = "/M_{}_seed{}.pkl".format(*saveinfo)
        pickle.dump(M, open(self.save_loc + Mfile, "wb"))
        Ufile = "/U_{}_seed{}.pkl".format(*saveinfo)
        pickle.dump(U, open(self.save_loc + Ufile, "wb"))
        Ifile = "/I_{}_seed{}.pkl".format(*saveinfo)
        pickle.dump(I, open(self.save_loc + Ifile, "wb"))
    
    def save_scaler(self):
        scaler_file = "/{}_seed{}.pkl".format(self.scaling, self.seed)
        savename = self.scaler_loc + scaler_file
        pickle.dump(self.scaler, open(savename, "wb"))

    def make_mask(self, data):
        mask = []
        for ele in data: 
            mask.append((~np.isnan(ele))*1.)
        return mask

    def make_utility_matrix(self, ydata):
        Umats = []
        for ydatum in ydata:
            Umats.append(self.umat_maker.make_utility_matrix(ydatum))
        return Umats
        
    def prep_scaler(self, Xdata):
        print("Preparing scaler ...")
        if self.scaling == "no_scale":
            self.scaler = None
        else:
            Xdata = np.concatenate(Xdata, axis=0)            
            self.scaler = load_scaler_function(self.scaling)
            self.scaler.fit(Xdata)
            self.save_scaler()


    def do_scaling(self, Xdata):
        if self.scaling == "no_scale":
            return Xdata
        else:
            Xtran = [self.scaler.transform(ele) for ele in Xdata]
            return Xtran


    def main(self):
        train_X, train_y, train_I = self.load_pickle("train")
        train_U = self.make_utility_matrix(train_y)
        train_X = self.preprocess_function(train_X)
        train_M = self.make_mask(train_X)
        self.prep_scaler(train_X)
        
        print("Working on train data ...")
        train_X = self.do_scaling(train_X)        
        train_X = self.imputation_function(train_X)
        self.save_pickle(train_X, train_y, train_M, train_U, train_I, "train")

        print("Working on valid data ...")
        valid_X, valid_y, valid_I = self.load_pickle("valid")
        valid_U = self.make_utility_matrix(valid_y) 
        valid_X = self.preprocess_function(valid_X)
        valid_M = self.make_mask(valid_X)        
        valid_X = self.do_scaling(valid_X)        
        valid_X = self.imputation_function(valid_X)
        self.save_pickle(valid_X, valid_y, valid_M, valid_U, valid_I, "valid")

        print("Working on test data ...")
        test_X, test_y, test_I = self.load_pickle("test")
        test_U = self.make_utility_matrix(test_y)
        test_X = self.preprocess_function(test_X)
        test_M = self.make_mask(test_X)        
        test_X = self.do_scaling(test_X)        
        test_X = self.imputation_function(test_X)
        self.save_pickle(test_X, test_y, test_M, test_U, test_I, "test")

        print("Working on mini data ...")
        mini_X, mini_y, mini_I = self.load_pickle("mini")
        mini_U = self.make_utility_matrix(mini_y)
        mini_X = self.preprocess_function(mini_X)
        mini_M = self.make_mask(mini_X)        
        mini_X = self.do_scaling(mini_X)        
        mini_X = self.imputation_function(mini_X)
        self.save_pickle(mini_X, mini_y, mini_M, mini_U, mini_I, "mini")

        
if __name__ == "__main__":
    import sys
    
    try: S = int(sys.argv[1])
    except: S = 1
    try: I = int(sys.argv[2])
    except: I = 0
    try: P = int(sys.argv[3])
    except: P = 4
    try: seed = int(sys.argv[4])
    except: seed = 1
    
    root = "../data"
    
    scaling = ["no_scale", "standard", "custom"][S]
    imputation = ["mean", "fwd"][I]
    preprocess = [None, "v1", "v2", "v3", "v4", "v5", "v6", "v7"][P]
    dc = DataConverter(root, scaling, imputation, preprocess, seed=seed)
    dc.main()
