import os, pickle, zipfile
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split as tts

class DatasetMaker(object):
    
    dirname = "physionet2019"
    raw_loc = "raw"
    procloc = "processed"

    def __init__(self, root, seed=1, split_data=True):
        self.data_loc = os.path.join(root, self.dirname, self.raw_loc)
        self.save_loc = os.path.join(root, self.dirname, self.procloc, "base")
        os.makedirs(self.save_loc, exist_ok=True)
        self.seed = seed

    def load_data(self):
        loc = os.path.join(self.data_loc, "*")
        files = sorted(glob(loc+"/*.psv"))
        if len(files) == 0:
            print("*"*80)
            print("Data not found")
            msg = "Please place challenge data at `{}` and unzip file"
            print(msg.format(self.data_loc))
            raise
        return np.array(files)
    
    def split_data(self, files):
        idx = np.arange(len(files))
        train, test = tts(idx, test_size=0.2, random_state=self.seed)
        valid, test = tts(idx[test], test_size=0.5, random_state=self.seed)
        return (files[train], train), (files[valid], valid), (files[test], test)

    def check_data(self, ydata):
        sepsis_loc = np.where(ydata==1)[0]
        if sepsis_loc.size == 0: return None
        last_match = sepsis_loc[-1] == len(ydata)-1

        diff = sepsis_loc[-1] - sepsis_loc[0] + 1
        length_match = len(sepsis_loc) == diff

        length = len(sepsis_loc)
        if not (last_match and length_match): 
            print("not matching")
            print(sepsis_loc)
        if length < 6:
            print("short label length")
            print(sepsis_loc)
        return None

    def make_Xy_idx(self, targets, datatype):
        targetfiles, idxs = targets
        X, y = [], []
        print("Working on {} ...".format(datatype))
        for target in tqdm(targetfiles):
            df = pd.read_csv(target, sep="|")
            _y = df.loc[:, "SepsisLabel"].values
            _X = df.drop("SepsisLabel", axis=1).values
            X.append(_X)
            y.append(_y)
        savename = self.save_loc+"/{}_{}_seed{}.pkl".format("{}", datatype,
                                                            self.seed)
        pickle.dump(X, open(savename.format("X"), "wb"))
        pickle.dump(y, open(savename.format("y"), "wb"))
        pickle.dump(idxs, open(savename.format("I"), "wb"))        

    def make_zip(self, testfiles, is_mini=False):
        print("Making zip file ...")
        if is_mini: 
            zip_name = self.save_loc + "/mini_seed{}.zip".format(self.seed)
        else: zip_name = self.save_loc + "/test_seed{}.zip".format(self.seed)
        output_zip = zipfile.ZipFile(zip_name, 'w')
        for testfile in tqdm(testfiles):
            output_zip.write(testfile)
        output_zip.close()
        
    def run(self):
        files = self.load_data()
        train, valid, test = self.split_data(files)
        self.make_Xy_idx(train, "train")
        self.make_Xy_idx(valid, "valid")
        self.make_Xy_idx(test, "test")
        self.make_Xy_idx((train[0][:100], train[1][:100]), "mini")
        self.make_zip(test[0])
        self.make_zip(train[0][:100], is_mini=True)

if __name__ == "__main__":
    import sys
    try: seed = int(sys.argv[1])
    except: seed = 1
    
    root = "../data"
    
    dm = DatasetMaker(root, seed=seed)
    dm.run()
