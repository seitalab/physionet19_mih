import torch
import torch.nn as nn

from functions.train_base_nn import BaseNNTrainer

class RNNTrainer(BaseNNTrainer):

    def convert(self, batch_data, mask):
        X, y, M, U = batch_data
        X = X.transpose(0, 1)
        y = y.transpose(0, 1)
        M = M.transpose(0, 1)
        U = U.transpose(0, 1)
        mask = mask.transpose(0, 1)
        batch_data = (X.float(), y.long(), M.float(), U.float())
        return batch_data, mask.float()
        
if __name__ == "__main__":
    import config
    from hyperparams import args
    from model_selector import load_model

    torch.manual_seed(args.seed)    
    model = load_model(args)

    trainer = RNNTrainer(model, args, config)
    trainer.run()
