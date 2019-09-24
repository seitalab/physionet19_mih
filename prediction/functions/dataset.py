import os, pickle
import torch
import numpy as np
from tqdm import tqdm

SHALLOWS = ["XGBClassifier", "RandomForestClassifier"]

class Dataset(object):

    cdata_loc = "../../concat" 
    
    def __init__(self, src_root, cvt_root, loadinfo,
                 datatype, seed, apply_softmax=False):
        assert datatype in ['train', 'valid', 'test', 'mini']

        if src_root is not None:
            self.src_root = os.path.expanduser(src_root)
        if cvt_root is not None:
            self.cvt_root = os.path.expanduser(cvt_root)

        if loadinfo is not None:
            loadtype, loadlocs = loadinfo
        else: loadtype, loadlocs = None, None
        
        self.apply_softmax = apply_softmax
        self.softmax_func = torch.nn.Softmax(dim=-1)
        
        self.seed_num = seed
        self.seed_info = "seed{}".format(seed)

        if self.dataset_exist(src_root, cvt_root, loadtype, datatype):
            I, X, y, U, M = self.load_dataset(src_root, cvt_root,
                                              loadtype, datatype)
        else:
            I, X, y, U, M = self.prep_data(src_root, cvt_root,
                                           datatype, loadlocs)
            if loadtype is not None:
                self.save_dataset(I, X, y, U, M, src_root, cvt_root,
                                  loadtype, datatype)
        idxs, data, label, umats, mask = I, X, y, U, M
                
        self.data, self.label = data, label
        self.mask, self.umats = mask, umats
        self.idxs = idxs

    def data_exist(self, loc, datatype, char):
        target = "{}/{}_{}_{}.pkl".format(loc, char, datatype, self.seed_info)
        return os.path.exists(target)
        
    def dataset_exist(self, src_root, cvt_root, loadtype, datatype):
        if loadtype is None: return False
        if src_root is not None:
            loc = os.path.join(src_root, self.cdata_loc, loadtype)
        else:
            loc = os.path.join(cvt_root, self.cdata_loc, loadtype)
        Iexist = self.data_exist(loc, datatype, "I")        
        Xexist = self.data_exist(loc, datatype, "X")
        yexist = self.data_exist(loc, datatype, "y")
        Uexist = self.data_exist(loc, datatype, "U")
        Mexist = self.data_exist(loc, datatype, "M")
        exist = np.array([Iexist, Xexist, yexist, Uexist, Mexist])
        return exist.all()

    def load_dataset(self, src_root, cvt_root, loadtype, datatype):
        if src_root is not None:
            loc = os.path.join(src_root, self.cdata_loc, loadtype)
        else:
            loc = os.path.join(cvt_root, self.cdata_loc, loadtype)
        Xfile = 'X_{}_seed{}.pkl'.format(datatype, self.seed_num)
        yfile = 'y_{}_seed{}.pkl'.format(datatype, self.seed_num)
        Mfile = 'M_{}_seed{}.pkl'.format(datatype, self.seed_num)
        Ufile = 'U_{}_seed{}.pkl'.format(datatype, self.seed_num)
        Ifile = 'I_{}_seed{}.pkl'.format(datatype, self.seed_num)
        X = self.open_pickle(loc, Xfile)
        y = self.open_pickle(loc, yfile)
        M = self.open_pickle(loc, Mfile)
        U = self.open_pickle(loc, Ufile)
        I = self.open_pickle(loc, Ifile)
        return I, X, y, U, M

    def save_dataset(self, I, X, y, U, M, src_root, cvt_root,
                     loadtype, datatype):
        if src_root is not None:
            saveloc = os.path.join(src_root, self.cdata_loc, loadtype)
        else:
            saveloc = os.path.join(cvt_root, self.cdata_loc, loadtype)        
        os.makedirs(saveloc, exist_ok=True)
        
        Xfile = 'X_{}_seed{}.pkl'.format(datatype, self.seed_num)
        yfile = 'y_{}_seed{}.pkl'.format(datatype, self.seed_num)
        Mfile = 'M_{}_seed{}.pkl'.format(datatype, self.seed_num)
        Ufile = 'U_{}_seed{}.pkl'.format(datatype, self.seed_num)
        Ifile = 'I_{}_seed{}.pkl'.format(datatype, self.seed_num)
        self.save_pickle(X, saveloc+"/"+Xfile)
        self.save_pickle(y, saveloc+"/"+yfile)
        self.save_pickle(M, saveloc+"/"+Mfile)
        self.save_pickle(U, saveloc+"/"+Ufile)
        self.save_pickle(I, saveloc+"/"+Ifile)

    def save_pickle(self, data, savename):
        pickle.dump(data, open(savename, "wb"))

    def prep_data(self, src_root, cvt_root, datatype, loadlocs):

        if src_root is not None:
            print("Loading data from {}...".format(src_root))
            Is, Xs, ys, Us, Ms = self.prep_srcdata(datatype)
        if cvt_root is not None:
            print("Loading data from {}...".format(cvt_root))            
            Ic, Xc, yc, Uc, Mc = self.prep_cvtdata(loadlocs, datatype)

        if src_root is None:
            I, X, y, U, M = Ic, Xc, yc, Uc, Mc
        elif cvt_root is None:
            I, X, y, U, M = Is, Xs, ys, Us, Ms
        else:
            print("Concatenating data ...")
            I, X, y, U, M = self.concat_src_cvt((Is, Xs, ys, Us, Ms),
                                                (Ic, Xc, yc, Uc, Mc))
        return I, X, y, U, M

    def __getitem__(self, index):
        batch = (self.data[index], self.label[index],
                 self.mask[index], self.umats[index])
        return batch, self.idxs[index]

    def __len__(self):
        return len(self.data)

    def break_index(self, idxs, data):
        new_idxs = []
        for i in range(len(data)):
            for j in range(len(data[i])):
                new_idxs.append([idxs[i], j])
        return np.array(new_idxs)

    def break_sequence(self, data):
        data = np.concatenate(data)
        return data

    def concat_src_cvt(self, src_data, cvt_data):
        # All I, y, U must be same
        Is, Xs, ys, Us, Ms = src_data
        Ic, Xc, yc, Uc, Mc = cvt_data
        order_s = np.argsort(Is)
        order_c = np.argsort(Ic)
        I, X, y, U, M = Is[order_s], [], ys[order_s], Us[order_s], []
        for i in tqdm(range(len(order_s))):
            if not Ic[order_c][i] == Is[order_s][i]: raise
            if not (yc[order_c][i] == ys[order_s][i]).all(): raise
            if not (Uc[order_c][i] == Us[order_s][i]).all(): raise
            _X = np.concatenate([Xc[order_c][i], Xs[order_s][i]], axis=1)
            _M = np.concatenate([Mc[order_c][i], Ms[order_s][i]], axis=1)
            X.append(_X)
            M.append(_M)
        X, M = np.array(X), np.array(M)
        return Is, X, y, U, M

    def prep_cvtdata(self, loadlocs, datatype):
        # All I, y, U must be same
        idxs, data = None, None
        
        for loc in tqdm(loadlocs):
            if loc == "src": continue
            I, X, y, U = self.load_data(loc, datatype)
            order = np.argsort(I)
            model = loc.split("/")[0]
            if (self.apply_softmax and model not in SHALLOWS):
                X = self.calc_softmax(X)
                
            if data is None:
                data = X[order]
                idxs = I[order]
            else:
                if not (idxs == I[order]).all(): raise                
                data = self.concat_X(X, order, data)
        y, U = y[order], U[order]
        mask = np.array([np.ones_like(d) for d in data])
        return idxs, data, y, U, mask

    def prep_srcdata(self, datatype):
        Xfile = 'X_{}_seed{}.pkl'.format(datatype, self.seed_num)
        yfile = 'y_{}_seed{}.pkl'.format(datatype, self.seed_num)
        Mfile = 'M_{}_seed{}.pkl'.format(datatype, self.seed_num)
        Ufile = 'U_{}_seed{}.pkl'.format(datatype, self.seed_num)
        Ifile = 'I_{}_seed{}.pkl'.format(datatype, self.seed_num)
        X = self.open_pickle(self.src_root, Xfile)
        y = self.open_pickle(self.src_root, yfile)
        M = self.open_pickle(self.src_root, Mfile)
        U = self.open_pickle(self.src_root, Ufile)
        I = self.open_pickle(self.src_root, Ifile)
        return I, X, y, U, M

    def calc_softmax(self, X):
        newX = []
        for x in X:
            x = torch.from_numpy(x)
            x = self.softmax_func(x)
            x = x.detach().cpu().numpy()
            newX.append(x)
        return np.array(newX)

    def concat_X(self, X, order, data): 
        for i in range(len(data)):
            data[i] = np.concatenate([data[i], X[order][i]], axis=-1)
        return data

    def load_data(self, loc, datatype):
        dataloc = os.path.join(self.cvt_root, self.seed_info, loc)
        Xfile = 'Xpred_{}.pkl'.format(datatype)
        yfile = 'y_{}.pkl'.format(datatype)
        Ufile = 'U_{}.pkl'.format(datatype)
        Ifile = 'I_{}.pkl'.format(datatype)
        X = self.open_pickle(dataloc, Xfile)
        y = self.open_pickle(dataloc, yfile)
        U = self.open_pickle(dataloc, Ufile)
        I = self.open_pickle(dataloc, Ifile)
        return I, X, y, U
        
    def open_pickle(self, dataloc, filename):
        file_loc = os.path.join(dataloc, filename)
        data = pickle.load(open(file_loc, "rb"))
        return np.array(data)

if __name__ == '__main__':
    pass
