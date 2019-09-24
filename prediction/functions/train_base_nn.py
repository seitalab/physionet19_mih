import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
from torch.nn.utils import clip_grad_value_
import adabound

import numpy as np
from tqdm import tqdm
from collections import defaultdict
from tensorboardX import SummaryWriter

from functions.train_base import BaseTrainer
from functions.utils_load import LoadingUtilities
from functions.batchgenerator import BatchGenerator
from functions.score_calculator import ScoreCalculator


class BaseNNTrainer(BaseTrainer, LoadingUtilities):
            
    def initial_process(self):
        self.save_dir = self.prepare_savedir(self.modelname, with_date=True)
        self.save_condition(self.modelname)
        self.save_architecture()
        self.prep_dataloc_path()
        
        self.model.to(self.args.device)
        self.tflog_dir = self.prepare_tflog_dir()
        
    def prep_dataloader(self, datatype, is_eval=False):
        print("Preparing {} dataloader".format(datatype))
        dataset = self.prep_dataset(datatype)
        if datatype == "train":
            split_mode = self.args.split_mode
        else: split_mode = "v1"

        if is_eval:
            shuffle = False
            split_mode = "v1"            
        else: shuffle = True
        
        loader = BatchGenerator(dataset, batch_size=self.args.batch_size, 
                                shuffle=shuffle, split_mode=split_mode,
                                device=self.args.device, seed=self.args.seed)
        return loader

    def convert(self, batch_data, mask):
        raise NotImplementedError
        
    def calc_loss(self, Xpred, y, U, mask):
        y = y.long()
        if self.model.training:
            u_val = self.u_calc.utility(Xpred, y, U, mask, weight=self.args.cw)
            u_val = u_val.sum()
        else:
            u_val = self.u_calc.utility(Xpred, y, U, mask, calc_grad=False)
        return u_val

    def train(self, epoch, iterator):
        train_loss, datanum = 0, 0        
        self.model.train()
        
        for _, batch, mask in tqdm(iterator):
            self.optimizer.zero_grad()
            
            datanum += batch[-1].size(0)

            (X, y, M, U), mask = self.convert(batch, mask)
            Xpred = self.model(X, M, mask)
            
            loss = self.calc_loss(Xpred, y, U, mask)
            loss.backward()
            
            if self.args.clip_value:
                clip_grad_value_(self.model.parameters(), self.args.clip_value)
                
            if self.args.optim == "bfgs":
                def closure():
                    self.optimizer.zero_grad()
                    Xpred = self.model(X, M, mask)
                    loss = self.calc_loss(Xpred, y, U, mask)
                    loss.backward()
                    return loss
                self.optimizer.step(closure)
            else: self.optimizer.step()
            if self.args.scheduler in ["cyclic"]: self.scheduler.step()
            
            train_loss += float(loss) * self.args.loss_scale
            
        train_loss = train_loss/datanum
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return {"train_loss": train_loss}

    def evaluate(self, epoch, iterator, store_mode=False):
        print("Validation")
        if isinstance(iterator, tuple): iterator = iterator[0]
        eval_loss, datanum, utility = 0, 0, []
        self.model.eval()
        if store_mode: store_data = []
        
        with torch.no_grad():
            for I, batch, mask in tqdm(iterator):
                datanum += batch[-1].size(0)
                (X, y, M, U), mask_ = self.convert(batch, mask)
                Xpred = self.model(X, M, mask_)
                
                u_value, u_loss = self.calc_loss(Xpred, y, U, mask_)
                eval_loss += float(u_loss)
                utility.append(u_value)
                if store_mode:
                    store_data = self.store(store_data, Xpred, I, batch, mask)

        utility = np.concatenate(utility).sum(axis=0)
        print("raw_utility:", utility)

        utility = ((utility[0] - utility[1])/(utility[2] - utility[1]))
        if store_mode:
            return store_data, utility

        eval_loss /= (1.0*datanum)
        print_info = (epoch, eval_loss, utility)
        print_text = "Epoch: {} Valid U score: {:.4f}, Utility: {:.4f}"
        print(print_text.format(*print_info))
        return {"eval_loss": eval_loss, "Utility": utility}

    def store(self, store_data, Xpred, I, batch, mask):
        mask = mask.detach().cpu().numpy()
        lens = mask.sum(axis=1).astype(int)
        Xpred = Xpred.detach().cpu().numpy()
        Xpred = np.swapaxes(Xpred, 0, 1)
        _, y, _, U = batch
        y = y.detach().cpu().numpy()
        U = U.detach().cpu().numpy()
        _Xpred, _y, _U = Xpred, y, U
        Xpred, y, U = [], [], []
        for i in range(len(lens)):
            Xpred.append(_Xpred[i, :lens[i]])
            y.append(_y[i, :lens[i]])
            U.append(_U[i, :lens[i]])
        I = list(I)
        if store_data == []: return [Xpred, y, U, I]
        Xpred = store_data[0] + Xpred
        y = store_data[1] + y
        U = store_data[2] + U
        I = store_data[3] + I
        return [Xpred, y, U, I]

    def prep_train_settings(self):
        self.prep_optim()
        self.prep_criterion()
        if self.args.scheduler is not None: self.prep_scheduler()

    
    def prep_scheduler(self):
        if self.args.scheduler == "step":
            self.scheduler = scheduler.StepLR(self.optimizer, step_size=50)
        elif self.args.scheduler == "exp":
            self.scheduler = scheduler.ExponentialLR(self.optimizer,
                                                     gamma=0.999)
        elif self.args.scheduler == "cyclic":
            self.scheduler = scheduler.CyclicLR(self.optimizer,
                                                step_size_up=5000,
                                                base_lr=0.1*self.args.lr,
                                                max_lr=self.args.lr)
            # Originally step_size_up: 2000
        elif self.args.scheduler == "plateau":
            self.scheduler = scheduler.ReduceLROnPlateau(self.optimizer)
        else:
            print("Scheduler not available: {}".format(self.args.scheduler))
            raise
        
    def prep_optim(self):
        if self.args.optim == "adam":
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.args.lr)
        elif self.args.optim == "adabound":
            self.optimizer = adabound.AdaBound(self.model.parameters(),
                                               lr=self.args.lr)
        elif self.args.optim == "rmsprop":
            self.optimizer = optim.RMSprop(self.model.parameters(),
                                           lr=self.args.lr)
        elif self.args.optim == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.args.lr)
        elif self.args.optim == "bfgs":
            self.optimizer = optim.LBFGS(self.model.parameters(),
                                         lr=self.args.lr)
        elif self.args.optim == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(),
                                         lr=self.args.lr)
        elif self.args.optim == "asgd":
            self.optimizer = optim.ASGD(self.model.parameters(),
                                        lr=self.args.lr)
        else:
            print("Invalid optimizer chosen")
            raise
        
    def prep_criterion(self):
        self.u_calc = ScoreCalculator(ufunc=self.args.ufunc,
                                      loss_scale=self.args.loss_scale)
        self.probfunc = torch.nn.LogSoftmax(dim=-1)        

    def store_final_result(self):
        weight_file = self.load_weight(self.save_dir)
        self.model.load_state_dict(torch.load(weight_file))
        self.model.to(self.args.device)

        result = {}
        train_loader = self.prep_dataloader("train", is_eval=True)
        store_train, us_train = self.evaluate("X", train_loader, True)
        result["train"] = us_train
        Xp_train, ytrain, Utrain, Itrain = store_train
        self.store_result(Xp_train, ytrain, Utrain, Itrain, "train")

        valid_loader = self.prep_dataloader("valid")
        store_valid, us_valid = self.evaluate("X", valid_loader, True)
        result["valid"] = us_valid
        Xp_valid, yvalid, Uvalid, Ivalid = store_valid
        self.store_result(Xp_valid, yvalid, Uvalid, Ivalid, "valid")

        test_loader = self.prep_dataloader("test")
        store_test, us_test = self.evaluate("X", test_loader, True)
        result["test"] = us_test
        Xp_test, y_test, U_test, I_test = store_test
        self.store_result(Xp_test, y_test, U_test, I_test, "test")
        self.save_score(result)
        
    def run(self, run_mini=False):

        if run_mini:
            train_loader = self.prep_dataloader("mini")
            valid_loader = self.prep_dataloader("mini")
        else:
            train_loader = self.prep_dataloader("train")
            valid_loader = self.prep_dataloader("valid")
        
        self.prep_train_settings()
        loss_log = defaultdict(float)

        writer = SummaryWriter(self.tflog_dir)
        for epoch in range(1, self.args.epochs+1):
            train_loss = self.train(epoch, train_loader)
            for key, value in train_loss.items():
                writer.add_scalar(key, value, epoch)
            if epoch % self.args.save_every != 0: continue

            print("-"*80)
            print("Train data evaluation")
            train_utility = self.evaluate(epoch, train_loader)["Utility"]
            writer.add_scalar("Train Utility", train_utility, epoch)
            print("-"*80)
            valid_loss = self.evaluate(epoch, valid_loader)
            loss_log[epoch] = valid_loss["Utility"]
            for key, value in valid_loss.items():
                writer.add_scalar(key, value, epoch)
            writer.add_scalar("Utility Gap",
                              train_utility - valid_loss["Utility"], epoch)
            self.save_model_nn(epoch, valid_loss["Utility"])
            print("-"*80)
            if self.args.scheduler in ["step", "exp"]: self.scheduler.step()
            elif self.args.scheduler in ["plateau"]:
                self.scheduler.step(valid_loss["Utility"])
        writer.close()
        self.save_bestmodel(loss_log)
        self.store_final_result()
        print("-"*80)
        

if __name__ == "__main__":
    pass
