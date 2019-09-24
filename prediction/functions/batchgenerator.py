import torch, random
import numpy as np

import functions.utils_batchgen as utils

class BatchGenerator(object):

    def __init__(self, dataloader, batch_size, split_mode="v1", 
                 shuffle=True, device='cpu', seed=1):
        random.seed(seed)
        if split_mode in ["v2", "v3", "v4"]:
            np.random.seed(seed)
            
        self.split_mode = split_mode
        self.batchsize = batch_size
        self.shuffle = shuffle
        self.device = device
        self.loader = dataloader        
        self.initialize()

    def initialize(self):
        if self.shuffle: self.shuffle_data()
        self.itercount = 0
        
    def shuffle_data(self):
        idxs = np.arange(len(self.loader))
        random.shuffle(idxs)
        self.loader.data = self.loader.data[idxs]
        self.loader.mask = self.loader.mask[idxs]
        self.loader.label = self.loader.label[idxs]
        self.loader.umats = self.loader.umats[idxs]
        self.loader.idxs = self.loader.idxs[idxs]
        
    def numpy_to_torch(self, data):
        data = torch.from_numpy(data)
        try: data = data.to(self.device)
        except: pass
        return data
        
    def split_data(self, batch_data):
        if self.split_mode == "v1": 
            batch_data, mask = utils.pad_batch(batch_data)
            batch_data = [self.numpy_to_torch(data) for data in batch_data]
            mask = self.numpy_to_torch(mask)
        elif self.split_mode == "v2":
            batch_data = utils.random_cut(batch_data)
            batch_data, mask = utils.pad_batch(batch_data)
            batch_data = [self.numpy_to_torch(data) for data in batch_data]
            mask = self.numpy_to_torch(mask)
        elif self.split_mode == "v3":
            batch_data = utils.random_insert(batch_data)
            batch_data, mask = utils.pad_batch(batch_data)
            batch_data = [self.numpy_to_torch(data) for data in batch_data]
            mask = self.numpy_to_torch(mask)
        elif self.split_mode == "v4":
            s = np.random.randint(3)
            if s == 0: batch_data = utils.random_insert(batch_data)
            if s == 1: batch_data = utils.random_cut(batch_data)
            batch_data, mask = utils.pad_batch(batch_data)
            batch_data = [self.numpy_to_torch(data) for data in batch_data]
            mask = self.numpy_to_torch(mask)
        elif self.split_mode == "v5":
            batch_data, mask = utils.cut_batch(batch_data, use_fw=False)
            batch_data = [self.numpy_to_torch(data) for data in batch_data]
            mask = self.numpy_to_torch(mask)
        elif self.split_mode == "v6":
            batch_data, mask = utils.cut_batch(batch_data, use_fw=True)
            batch_data = [self.numpy_to_torch(data) for data in batch_data]
            mask = self.numpy_to_torch(mask)
        else: raise

        return batch_data, mask
    
    def __len__(self):
        div = len(self.loader)/self.batchsize
        mod = len(self.loader)%self.batchsize
        return int(div + 1 * (mod > 0))

    def __iter__(self):
        return self    

    def __next__(self):
        if self.itercount < 0:
            self.initialize()
            raise StopIteration()
        idx_s = self.itercount * self.batchsize
        idx_e = idx_s + self.batchsize
        if idx_e >= len(self.loader):
            idx_e = len(self.loader)
            self.itercount = -1
        else:
            self.itercount += 1
        
        batch_data, idxs = self.loader[idx_s:idx_e]
        batch_data, mask = self.split_data(batch_data)
        return idxs, batch_data, mask

if __name__ == '__main__':
    from dataloader import DataLoader
    #loc = "../data/physionet2019/processed/zero_imp-standard"
    loc = "~/mnt/physionet2019/processed_pipe/zero_imp-standard"
    loader = DataLoader(loc, 'mini', seed=1, for_nonseq=True)
    bg = BatchGenerator(loader, batch_size=24)
    for ep in range(3):
        print("*"*80)
        print("Epoch: %s"%(ep+1))
        for idx, batch_data, m in bg:
            try:
                X1, X2, y, M1, M2, U = batch_data
                print(idx)
                print(X1.shape, X2.shape)
                print(y.shape, M1.shape, M2.shape, U.shape)
                print(m[0].shape, m[1].shape)
            except:
                X1, y, M1, U = batch_data
                print(idx)
                print(X1.shape)
                print(y.shape, M1.shape, U.shape)
                #print(m.shape)
            #break
        #break
