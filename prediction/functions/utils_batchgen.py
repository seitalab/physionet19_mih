import numpy as np

def pad_batch(batch_data):
    Xdata, ydata, Mdata, Udata = batch_data
    lengths = np.array([ele.shape[0] for ele in Xdata])
    max_len = lengths.max()
    pad_len = max_len - lengths
    mask = np.zeros([lengths.size, max_len])
    Xnew, ynew, Mnew, Unew = [], [], [], []
    for i in range(lengths.size):
        mask[i, :lengths[i]] = 1
        Xnew.append(add_blank(Xdata[i], pad_len[i]))
        ynew.append(add_blank(ydata[i], pad_len[i]))
        Mnew.append(add_blank(Mdata[i], pad_len[i]))
        Unew.append(add_blank(Udata[i], pad_len[i]))
    Xdata, ydata = np.array(Xnew), np.array(ynew)
    Mdata, Udata = np.array(Mnew), np.array(Unew)
    return (Xdata, ydata, Mdata, Udata), mask

def add_blank(data, padlen):
    if padlen == 0: return data
    if data.ndim == 1: pad = np.zeros([padlen])
    else: pad = np.zeros([padlen, data.shape[1]])

    data = np.concatenate([data, pad])
    return data

def random_cut(data, cut_len=50):
    Xnew, ynew, Mnew, Unew = [], [], [], []
    for i in range(len(data[0])):
        original_len = len(data[0][i])
        cut = max(5, int(np.random.normal(cut_len, 20)))
        cut = min(original_len, cut)
        xorg = data[0][i]
        xnew = data[0][i][-1*cut:]
        xnew[0, -1] = xorg[0, -1]
        Xnew.append(xnew)
        ynew.append(data[1][i][-1*cut:])
        Mnew.append(data[2][i][-1*cut:])
        Unew.append(data[3][i][-1*cut:])
    Xdata, ydata = np.array(Xnew), np.array(ynew)
    Mdata, Udata = np.array(Mnew), np.array(Unew)
    return (Xdata, ydata, Mdata, Udata)

def random_insert(data, maxlen=300):
    Xnew, Ynew, Mnew, Unew = [], [], [], []
    for i in range(len(data[0])):
        original_len = len(data[0][i])
        new_length = int(np.random.normal(300, 20))
        pad = new_length - original_len
        if pad > 0:
            duplicate_idxs = sorted(np.random.choice(original_len, pad))
            new_idxs = sorted(np.concatenate([np.arange(original_len),
                                              duplicate_idxs]))
            xnew = data[0][i][new_idxs]
            ynew = data[1][i][new_idxs]
            mnew = data[2][i][new_idxs]
            unew = data[3][i][new_idxs]
        else:
            xnew = data[0][i]
            ynew = data[1][i]
            mnew = data[2][i]
            unew = data[3][i]
        Xnew.append(xnew)
        Ynew.append(ynew)
        Mnew.append(mnew)
        Unew.append(unew)
    Xdata, ydata = np.array(Xnew), np.array(Ynew)
    Mdata, Udata = np.array(Mnew), np.array(Unew)
    return (Xdata, ydata, Mdata, Udata)

def cut_batch(data, use_fw=False):
    Xnew, ynew, Mnew, Unew = [], [], [], []
    cut = min([len(ele) for ele in data[0]])
    for i in range(len(data[0])):
        xorg = data[0][i]
        if use_fw:
            xnew = data[0][i][:cut]
            ynew.append(data[1][i][:cut])
            Mnew.append(data[2][i][:cut])
            Unew.append(data[3][i][:cut])
        else:
            xnew = data[0][i][-1*cut:]
            ynew.append(data[1][i][-1*cut:])
            Mnew.append(data[2][i][-1*cut:])
            Unew.append(data[3][i][-1*cut:])
        xnew[0, -1] = xorg[0, -1]
        Xnew.append(xnew)
    mask = np.ones([len(data[0]), cut])
    Xdata, ydata = np.array(Xnew), np.array(ynew)
    Mdata, Udata = np.array(Mnew), np.array(Unew)
    return (Xdata, ydata, Mdata, Udata), mask
