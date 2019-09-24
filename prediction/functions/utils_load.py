import os, pickle
import torch
import numpy as np
from glob import glob

class LoadingUtilities(object):

    def load_args(self, model_loc, device="cpu"):
        args = pickle.load(open(model_loc + "/model_info.pkl", "rb"))
        args.device = device
        if device != "cpu": self.model.to(device)
        args = self.prep_additional_args(args)
        self.args = args

    def prep_additional_args(self, args):
        try: args.ufunc
        except: args.ufunc = "v1"
        try: args.seq_w
        except: args.seq_w = "v0"
        return args

    def load_scaler(self):
        if self.config.scaling in ["standard"]:
            scalerfile = "_seed{}.pkl".format(self.seed)
            scalerfile = self.config.scaler_loc + self.config.scaling\
                         + scalerfile
            self.scaler = pickle.load(open(scalerfile, "rb"))
        else:
            raise

    def prepare_model(self, model_loc, epoch=None):
        print("Loading model from {} ...".format(model_loc))
        model = torch.load(model_loc+"/architecture")
        if epoch is None: state = glob(model_loc+"/best-*.pth")[0]
        else: state = glob(model_loc+"/ep{:04d}-*.pth".format(epoch))[0]
        print("Model: {}".format(state))
        model.load_state_dict(torch.load(state))
        self.model = model

    def select_weight(self, modelname, modelnum):
        seed_num = "seed{}".format(self.seed)
        modelsloc = os.path.join(self.config.model_dir, modelname, seed_num)
        model_dir = sorted(glob(modelsloc+"/*"))[modelnum]
        return model_dir

    # For loading and copying model
    def load_weight(self, load_dir):
        best_model = glob(load_dir+"/best-*.pth")
        if best_model: return best_model[0]
        models = sorted(glob(load_dir+"/ep*.pth"))
        if models == []: raise
        return models[-1]

    def load_model(self, weight_loc, copy_model=False):
        weight_file = self.load_weight(weight_loc)

        if copy_model:
            self.copy_basemodel(weight_file, self.save_dir)
            self.save_basemodel_info(weight_file, self.save_dir)

        print("Loading pretrained model weight from {} ...".format(weight_loc))
        self.model.load_state_dict(torch.load(weight_file))
        self.model.to(self.args.device)

    def save_basemodel_info(self, srcfile, save_loc):
        # Used when loading pretrained model
        savetxt = "source model:{}\n".format(srcfile)
        open(save_loc+"/source_modelinfo.txt", "w").write(savetxt)
        
    def copy_basemodel(self, srcfile, dst_dir):
        cmd = "cp {} {}/source_model.pth".format(srcfile, dst_dir)
        os.system(cmd)
        src_dir = os.path.dirname(srcfile)
        cmd = "cp {}/model_info.txt {}/source_info.txt".format(src_dir,
                                                               dst_dir)
        os.system(cmd)

    
