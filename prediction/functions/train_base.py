import os, pickle
import numpy as np

from functions.dataset import Dataset
from functions.utils_save import SavingUtilities

class BaseTrainer(SavingUtilities):

    def __init__(self, model, args, config, base_list):
    
        self.args = args
        self.model = model
        self.config = config

        self.modelname = str(self.model.__class__.__name__)
        self.base_list = base_list
        
        self.initial_process()
        
    def initial_process(self):
        self.save_dir = self.prepare_savedir(self.modelname)
        self.save_condition(self.modelname)
        self.prep_dataloc_path()

    def prep_dataset(self, datatype):
        dataset = Dataset(self.srcdata_loc, self.cvtdata_loc,
                          self.base_list, datatype, self.args.seed)
        return dataset
        
    def prep_dataloc_path(self):
        scaling = self.config.scaling[self.args.scaler]
        imputation = self.config.imputation[self.args.imp]
        preprocess = self.config.preprocess[self.args.preprocess]

        processtype = "{}-{}".format(imputation, scaling)
        if preprocess is not None: processtype += "-{}".format(preprocess)
        
        srcdata_loc = os.path.join(self.config.dataset_loc, processtype)
        
        if self.base_list is None:
            self.srcdata_loc = srcdata_loc
            self.cvtdata_loc = None
        elif "src" in self.base_list[1]:
            self.srcdata_loc = srcdata_loc
            self.cvtdata_loc = os.path.join(self.config.model_dir)
        else:
            self.srcdata_loc = None
            self.cvtdata_loc = os.path.join(self.config.model_dir)

    def reconst_seq(self, data, indexs):
        reconst = []
        ids = sorted(np.unique(indexs[:, 0]))
        for i in ids:
            target = np.where(indexs[:, 0]==i)[0]
            order = np.argsort(indexs[:, 1][target])
            reconst.append(data[target[order]])
        return np.array(reconst)

    def reconst_index(self, indexs):
        return sorted(np.unique(indexs[:, 0]))

    def train(self, Xdata, ydata):
        raise NotImplementedError
        
    def evaluate(self, Xdata, Udata):
        raise NotImplementedError
        
    def store_result(self, Xpred, ydata, Udata, Idata, datatype):
        print("Saving {} data result ...".format(datatype))
        savename = self.save_dir + "/{}_{}.pkl"
        pickle.dump(Idata, open(savename.format("I", datatype), "wb"))
        pickle.dump(ydata, open(savename.format("y", datatype), "wb"))
        pickle.dump(Udata, open(savename.format("U", datatype), "wb"))
        pickle.dump(Xpred, open(savename.format("Xpred", datatype), "wb"))

    def run(self, run_mini=False):
        raise NotImplementedError

    def save_score(self, scores):
        print("Score: ", scores)
        result = "train,valid,test\n{:04f},{:04f},{:04f}"
        result = result.format(scores["train"], scores["valid"], 
                               scores["test"])
        open(self.save_dir + "/score.txt", "w").write(result)

if __name__ == "__main__":
    pass
