import os, pickle
import torch
import numpy as np
from glob import glob
from datetime import datetime

class SavingUtilities(object):

    def prepare_tflog_dir(self):
        modelinfo = self.save_dir.split("/")
        tflog_dir = os.path.join(*[self.config.tflog_dir]+modelinfo[3:])
        return tflog_dir

    def get_timestamp(self):
        timestamp = datetime.now()
        timestamp = timestamp.strftime('%Y%m%d-%H%M%S')[2:]
        return timestamp
        
    def prepare_savedir(self, modelname, with_date=False):
        seed_num = "seed{}".format(self.args.seed)
        param_info = paraminfo_preparator(self.args, modelname)
        save_dir = os.path.join(self.config.model_dir, seed_num,
                                modelname, param_info)
        if with_date:
            save_dir = os.path.join(save_dir, self.get_timestamp())
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def save_condition(self, modelname):
        save_txt = "%s\n"%modelname
        for key, value in self.args.__dict__.items():
            save_txt += "\t%s: %s\n"%(key, value)
        open(self.save_dir+'/model_info.txt', 'w').write(save_txt)
        pickle.dump(self.args, open(self.save_dir+'/model_info.pkl', 'wb'))
        print("Saved model setting")

    def save_architecture(self):
        print("Saving model architecture ...")
        model = self.model.to("cpu")
        torch.save(model, self.save_dir+"/architecture")

    def save_model_nn(self, epoch, valloss):
        epoch = self.config.ep_str.format(epoch)
        val_loss = '{:.4f}'.format(valloss)
        modelfile = self.config.modelfile.format(epoch, val_loss)
        
        filename  = os.path.join(self.save_dir, modelfile)
        torch.save(self.model.state_dict(), filename)
        print("Saved model as {}".format(filename))

    def save_model_sk(self):
        savename = self.save_dir + "/model.pkl"
        pickle.dump(self.model, open(savename, "wb"))

    def save_bestmodel(self, loss_log):
        #best_idx = np.argmin(list(loss_log.values()))
        best_idx = np.argmax(list(loss_log.values()))
        best_epoch = list(loss_log.keys())[best_idx]
        best_epoch = self.config.ep_str.format(best_epoch)
        model_file = self.config.modelfile.format(best_epoch, "*")
        model_file = os.path.join(self.save_dir, model_file)
        model_file = glob(model_file)[0]
        best_file  = model_file.replace(best_epoch, "best")
        os.system("cp {} {}".format(model_file, best_file))
        print("Saved best model ({})".format(model_file))
    
def paraminfo_preparator(args, modelname):
    info_common = (args.base_list, args.scaler, args.imp, args.preprocess)
    param_common = "Blist-{}_Sc-{:02d}_Imp-{:02d}_Pp-{:02d}_"
    param_common = param_common.format(*info_common)
    if (modelname[:3] == "GRU" or modelname[:6] == "CNNGRU"):
        info1 = (args.batch_size, args.lr, args.clip_value, args.cw)
        info2 = (args.epochs, args.h_dim, args.optim, args.scheduler)
        info3 = (args.loss_scale, args.split_mode, args.ufunc)
        param_text1 = "bs-{}_lr-{}_clip-{}_cw-{}_".format(*info1)
        param_text2 = "ep-{}_hdim-{}_optim-{}_sch-{}_".format(*info2)
        param_text3 = "lscale-{}_split-{}_ufunc-{}".format(*info3)
        param_text = param_common + param_text1 + param_text2 + param_text3
        return param_text
    else:
        raise
