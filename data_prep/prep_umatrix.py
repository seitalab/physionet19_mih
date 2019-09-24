import os, pickle
import numpy as np

class UtilityMatrixMaker(object):

    def __init__(self, dt_early=-12, dt_optimal=-6, dt_late=3.0,
                 max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0):
        self.dt_early = dt_early
        self.dt_optimal = dt_optimal
        self.dt_late = dt_late
        self.u_fp = u_fp
        self.u_tn = u_tn
        self.form_utility_slope(max_u_tp, min_u_fn)

    def form_utility_slope(self, max_u_tp, min_u_fn):
        self.m_1 = float(max_u_tp) / float(self.dt_optimal - self.dt_early)
        self.b_1 = -self.m_1 * self.dt_early
        self.m_2 = float(-max_u_tp) / float(self.dt_late - self.dt_optimal)
        self.b_2 = -self.m_2 * self.dt_late
        self.m_3 = float(min_u_fn) / float(self.dt_late - self.dt_optimal)
        self.b_3 = -self.m_3 * self.dt_optimal
    
    def make_utility_matrix(self, label):
        """
        Input(label data): [seqlen]
        Output(Utility score): [seqlen, 2]
        """
        score_matrix = np.ones([len(label), 2])
        score_matrix[:, 1] = self.u_fp
        score_matrix[:, 0] = self.u_tn

        if not label.any(): return score_matrix

        t_sepsis = np.argmax(label) - self.dt_optimal
        for t in range(len(label)):
            before_early = (t < t_sepsis + self.dt_optimal)
            before_optimal = (t <= t_sepsis + self.dt_optimal)
            before_late = (t <= t_sepsis + self.dt_late)
            before_late = bool(before_late * (1 - before_optimal))
            
            slope1 = self.m_1 * (t - t_sepsis) + self.b_1
            slope2 = self.m_2 * (t - t_sepsis) + self.b_2
            slope3 = self.m_3 * (t - t_sepsis) + self.b_3
            fp_mat = (np.ones_like(slope1)*self.u_fp)
            pscore = np.maximum(slope1, fp_mat)
            b_opt_0 = float(before_optimal) * pscore
            b_late0 = float(before_late) * slope2
            b_late1 = float(before_late) * slope3
            
            score_matrix[t, 1] = b_opt_0 + b_late0
            score_matrix[t, 0] = 0 + b_late1
        return score_matrix
        
if __name__ == "__main__":
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 
                       1, 1, 1, 1, 1, 1, 0, 0, 0])
    umm = UtilityMatrixMaker()
    u = umm.make_utility_matrix(labels)
    print(u)

    labels = np.array([0, 0, 0, 0, 0, 0])
    u = umm.make_utility_matrix(labels)
    print(u)

    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0])
    u = umm.make_utility_matrix(labels)
    print(u)
