import torch
from importlib import import_module
from functions.train_rnn import RNNTrainer
from modellist import *

class Executer(object):

    def __init__(self, args, config):
        torch.manual_seed(args.seed)
        
        self.args = args
        self.config = config
        
        modelname = args.model
        
        base_list = self.prep_baselist(args.base_list)
        if base_list is not None: _base_list = base_list[1]
        else: _base_list = base_list
        self.set_input_dim(_base_list)

        model = self.load_model(modelname, base_list)
        self.trainer = self.load_trainer(model, base_list)


    def set_input_dim(self, base_list):
        if base_list is None: pass
        elif "src" in base_list:
            self.args.i_dim = int(len(base_list)*2) + self.args_i_dim - 2
        else:
            self.args.i_dim = int(len(base_list)*2)

    def prep_baselist(self, base_list):
        if base_list is None: return None
        elif base_list == "l1": return ("l1", l1)
        elif base_list == "l1s": return ("l1s", l1 + ["src"])

    def load_trainer(self, model, base_list):
        Trainer = RNNTrainer(model, self.args, self.config, base_list)
        return Trainer

    def load_model(self, modelname, base_list):
        modelfile = "architectures.{}".format(modelname)
        ModelClass = import_module(modelfile)
        Model = ModelClass.__dict__[modelname]
        model = Model(self.args, base_list)
        print("Loaded {} ...".format(ModelClass.__name__))        
        return model
        
    def execute(self, mini=False):
        self.trainer.run(mini)

if __name__ == "__main__":
    import config    
    from hyperparams import args

    executer = Executer(args, config)
    executer.execute()
