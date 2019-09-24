import torch
import numpy as np

def mod_utility(batch_utility, ufunc, valid_state):
    if ufunc == "v1":
        return (batch_utility + 2)/3.
    elif ufunc == "v0":
        return batch_utility
    elif ufunc == "v2":
        # Always predict 0
        batch_utility[:,:,0] = (batch_utility[:,:,0] + 1)/2. # U_n
        zero_mat = torch.zeros_like(batch_utility[:,:,0])
        batch_utility[:,:,0] = torch.max(batch_utility[:,:,0], zero_mat)
        batch_utility[:,:,1] = (batch_utility[:,:,1] + 0.05)/(1. + 0.05) # U_p
        return batch_utility
    elif ufunc == "v3":
        # Always predict near 0
        batch_utility[:,:,1] = (batch_utility[:,:,1] > 0).float()
        batch_utility[:,:,0] = (batch_utility[:,:,0] + 2)/5.
        return batch_utility
    elif ufunc == "v4":
        batch_utility[:,:,1] = (batch_utility[:,:,1] > 0).float()
        batch_utility[:,:,0] = 0.1
        return batch_utility
    elif ufunc == "v5":
        batch_utility[:,:,1] = (batch_utility[:,:,1] > 0).float()
        batch_utility[:,:,0] = (batch_utility[:,:,0] + 2)/30.
        return batch_utility
    elif ufunc == "v6":
        batch_utility[:,:,1] = (batch_utility[:,:,1] > 0).float()
        batch_utility[:,:,0] = 0.01
        return batch_utility
    elif ufunc == "v7":
        batch_utility[:,:,1] = (batch_utility[:,:,1] > 0).float()
        batch_utility[:,:,0] = (batch_utility[:,:,0] + 2)/50.
        return batch_utility
    elif ufunc == "v8":
        batch_utility[:,:,1] = (batch_utility[:,:,1] + 2)/3.
        batch_utility[:,:,0] = (batch_utility[:,:,0] + 2)/50.
        return batch_utility
    elif ufunc == "v9":
        batch_utility[:,:,1] = (batch_utility[:,:,1] > 0).float()
        batch_utility[:,:,0] = (batch_utility[:,:,0] + 2)/100.
        return batch_utility
    elif ufunc == "v10":
        batch_utility[:,:,1] = (batch_utility[:,:,1] > 0).float()
        batch_utility[:,:,0] = (batch_utility[:,:,0] + 2)/75.
        return batch_utility
    elif ufunc == "v11":
        if valid_state is None: valid_state = 0
        
        if valid_state < 50: denom = 50. 
        elif valid_state < 100: denom = 75.
        else: denom = 100.
        batch_utility[:,:,1] = (batch_utility[:,:,1] > 0).float()
        batch_utility[:,:,0] = (batch_utility[:,:,0] + 2)/denom
        return batch_utility
    elif ufunc == "v12":
        if valid_state is None: valid_state = (1, 1)
        denom = 75 * (valid_state[1]/valid_state[0]) # neg/pos
        batch_utility[:,:,1] = (batch_utility[:,:,1] > 0).float()
        batch_utility[:,:,0] = (batch_utility[:,:,0] + 2)/denom
        return batch_utility
    elif ufunc == "v13":
        if valid_state is None: valid_state = (1, 1, 1)
        min_decay, ep_max = 0.8, 250
        decay = 1 - ((1 - min_decay)*(min(ep_max, valid_state[2])/ep_max))
        denom = 50 * (valid_state[1]/valid_state[0]) * decay # neg/pos
        batch_utility[:,:,1] = (batch_utility[:,:,1] > 0).float()
        batch_utility[:,:,0] = (batch_utility[:,:,0] + 2)/denom
        return batch_utility
    elif ufunc == "v14":
        if valid_state is None: valid_state = (1)
        min_decay, ep_max = 0.8, 250
        decay = 1 - ((1 - min_decay)*(min(ep_max, valid_state)/ep_max))
        denom = 50 * decay
        batch_utility[:,:,1] = (batch_utility[:,:,1] > 0).float()
        batch_utility[:,:,0] = (batch_utility[:,:,0] + 2)/denom
        return batch_utility
    elif ufunc == "v15":
        if valid_state is None: valid_state = (1)

        if valid_state > 150:
            return (batch_utility + 2)/3.
        batch_utility[:,:,1] = (batch_utility[:,:,1] > 0).float()
        batch_utility[:,:,0] = (batch_utility[:,:,0] + 2)/50.
        return batch_utility
    elif ufunc == "v16":
        batch_utility[:,:,1] = (batch_utility[:,:,1] > 0).float()
        batch_utility[:,:,0] = (batch_utility[:,:,0] + 2)/40.
        return batch_utility
    elif ufunc == "v17":
        batch_utility[:,:,1] = (batch_utility[:,:,1] > 0).float()
        batch_utility[:,:,0] = (batch_utility[:,:,0] + 1)/20.
        return batch_utility
    elif ufunc == "v18":
        batch_utility[:,:,1] = (batch_utility[:,:,1] > 0).float()
        batch_utility[:,:,0] = (batch_utility[:,:,0])/20.
        return batch_utility
    elif ufunc == "v19":
        batch_utility[:,:,1] = (batch_utility[:,:,1] > 0).float()
        batch_utility[:,:,0] = (batch_utility[:,:,0] + 2)/60.
        return batch_utility
