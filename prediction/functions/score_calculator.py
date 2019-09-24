import torch
import numpy as np

from functions.ufuncs import mod_utility

class ScoreCalculator(object):

    def __init__(self, ufunc, loss_scale=1):
        self.ufunc = ufunc
        self.loss_scale = loss_scale
        self.valid_state = None
        if self.ufunc == "v0":
            self.function = torch.nn.Softmax(dim=-1)
        else: self.function = torch.nn.LogSoftmax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)
        
    def form_select_matrix(self, prediction):
        select = torch.zeros_like(prediction).view(-1, 2)
        _prediction = prediction.view(-1, 2)
        pred = torch.max(_prediction, dim=-1)[1].view(-1)
        p_idx = pred.nonzero().view(-1)
        n_idx = (pred==0).nonzero().view(-1)
        select[p_idx, 1] = 1
        select[n_idx, 0] = 1
        return select.view(prediction.size())

    def update_progress(self, **kwargs):
        if self.ufunc in ["v11", "v14", "v15"]:
            self.valid_state = kwargs["epoch"]
        elif self.ufunc == "v12":
            pos, neg = kwargs["acc_positive"], kwargs["acc_negative"]
            self.valid_state = (pos, neg)
        elif self.ufunc == "v13":
            raise
            
    def utility(self, pred_batch, label_batch, U_batch, mask_batch,
                weight=1, calc_grad=True):
        pred_probs = self.softmax(pred_batch)
        pred_batch = self.function(pred_batch)
        expand_dim = tuple(mask_batch.size())+(2,)
        mask_batch = mask_batch.unsqueeze(-1).expand(expand_dim)

        if calc_grad:
            utility = mod_utility(U_batch, self.ufunc, self.valid_state)
            score = utility * pred_batch * mask_batch
            c_weight = torch.ones_like(score)
            c_weight[:,:,1] = weight
            score = score * c_weight
            score = -1 * (score.sum()/(weight+1))
            #return score
            return score/self.loss_scale
            #return score/10.
        else:
            optimal, _ = torch.max(U_batch*mask_batch, dim=-1)

            basemat = U_batch[:, :, 0]
            basemat = basemat * mask_batch[:,:,0]
            
            utility = U_batch * mask_batch
            select = self.form_select_matrix(pred_batch)
            score = select * utility

            uloss = utility * pred_probs
            #check_pred(utility, pred_probs, uloss)
            uloss = uloss.sum()
            uloss = uloss.detach().cpu().numpy()
            
            u_opt = optimal.sum(dim=0)
            ubase = basemat.sum(dim=0)
            score = score.sum(dim=0).sum(dim=-1)
            return self.convert_to_numpy(score, ubase, u_opt), uloss

    def convert_to_numpy(self, score, ubase, u_opt):
        score = score.detach().cpu().numpy()
        ubase = ubase.detach().cpu().numpy()
        u_opt = u_opt.detach().cpu().numpy()
        return np.stack([score, ubase, u_opt]).T


def check_pred(utility, pred_probs, uloss):
    V = 1
    a = utility[:,V].sum(dim=-1).cpu().detach().numpy()
    idxs = np.where(a==0)[0]
    if idxs[0] + 1 == idxs[1]: idx = idxs[0]
    else: idx = idxs[1]
    print(utility[:idx,V])
    input()
    print(pred_probs[:idx,V])
    input()
    print(uloss[:idx, V])
    input()
    
if __name__ == "__main__":
    pass
