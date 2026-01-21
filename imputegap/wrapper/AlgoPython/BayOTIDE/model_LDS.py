import numpy as np

# import scipy
import torch

# import utils
"""
base model of LDS system

"""


# SDE-inference of LDS system
class LDS_GP:

    def __init__(self, hyper_para_dict):
        self.device = hyper_para_dict["device"]  # add the cuda version later

        self.F = hyper_para_dict["F"].double().to(
            self.device)  # transition mat-SDE
        self.H = hyper_para_dict["H"].double().to(
            self.device)  # emission mat-SDE
        self.R = hyper_para_dict["R"].double().to(
            self.device)  # emission noise

        self.P_inf = hyper_para_dict["P_inf"].double().to(self.device)  # P_inf

        self.fix_int = hyper_para_dict[
            "fix_int"]  # whether the time interval is fixed

        if self.fix_int:
            self.A = torch.matrix_exp(self.F * self.fix_int)
            self.Q = self.P_inf - torch.mm(torch.mm(self.A, self.P_inf),self.A.T)
        else:
            # pre-compute and store the transition-mats for dynamic gap
            # we can also do this in each filter predict-step, but relative slow
            """check whether compute A,Q here or at  predict-step"""

            self.time_int_list = hyper_para_dict["time_int_list"].to(
                self.device)

            # self.A_list = [torch.matrix_exp(self.F*time_int).double() for time_int in self.time_int_list]
            # self.Q_list = [self.P_inf - torch.mm(torch.mm(A,self.P_inf),A.T) for A in self.A_list]

        self.m_0 = hyper_para_dict["m_0"].double().to(self.device)  # init mean
        self.P_0 = hyper_para_dict["P_0"].double().to(self.device)  # init var

        self.reset_list()

    def reset_list(self):
        self.m = self.m_0  # store the current state(mean)
        self.P = self.P_0  # store the current state(var)

        self.m_update_list = []  # store the filter-update state(mean)
        self.P_update_list = []  # store the filter-update state(var)

        self.m_pred_list = []  # store the filter-pred state(mean)
        self.P_pred_list = []  # store the filter-pred state(var)

        self.m_smooth_list = []  # store the smoothed state(mean)
        self.P_smooth_list = []  # store the smoothed state(mean)

    def reset_smooth_list(self):
        self.m_smooth_list = []  # store the smoothed state(mean)
        self.P_smooth_list = []  # store the smoothed state(mean)

    def filter_predict(self, ind=None, time_int=None):
        if self.fix_int:
            self.m = torch.mm(self.A, self.m).double()
            self.P = torch.mm(torch.mm(self.A, self.P), self.A.T) + self.Q

            self.m_pred_list.append(self.m)
            self.P_pred_list.append(self.P)

        # none-fix-interval, recompute A,Q based on current time-interval
        else:

            if ind is None:
                raise Exception(
                    "need to input the state-index for non-fix-interval case")

            time_int = self.time_int_list[ind]
            self.A = torch.matrix_exp(self.F * time_int).double()
            self.Q = self.P_inf - torch.mm(torch.mm(self.A, self.P_inf),
                                           self.A.T)

            # self.A = self.A_list[ind]
            # self.Q = self.Q_list[ind]

            self.m = torch.mm(self.A, self.m).double()
            self.P = torch.mm(torch.mm(self.A, self.P), self.A.T) + self.Q

            self.m_pred_list.append(self.m)
            self.P_pred_list.append(self.P)

            # self.tau_list.append(tau) # store all the time interval

    def filter_update(self, y, R=None, add_to_list=True):

        if R is None:
            R = self.R

        V = y - torch.mm(self.H, self.m)
        S = torch.mm(torch.mm(self.H, self.P), self.H.T) + R
        K = torch.mm(torch.mm(self.P, self.H.T), torch.linalg.pinv(S))
        self.m = self.m + torch.mm(K, V)
        self.P = self.P - torch.mm(torch.mm(K, self.H), self.P)

        if add_to_list:
            self.m_update_list.append(self.m)
            self.P_update_list.append(self.P)

    def smooth(self):

        # start from the last end

        self.N_time = len(self.m_update_list)

        m_s = self.m_update_list[-1]
        P_s = self.P_update_list[-1]

        self.m_smooth_list.insert(0, m_s)
        self.P_smooth_list.insert(0, P_s)

        if self.fix_int:
            for i in reversed(range(self.N_time - 1)):

                m = self.m_update_list[i]
                P = self.P_update_list[i]

                m_pred = self.m_pred_list[i + 1]
                P_pred = self.P_pred_list[i + 1]

                G = torch.mm(torch.mm(P, self.A.T), torch.linalg.pinv(P_pred))
                m_s = m + torch.mm(G, m_s - m_pred)
                P_s = P + torch.mm(torch.mm(G, P_s - P_pred), G.T)

                self.m_smooth_list.insert(0, m_s)
                self.P_smooth_list.insert(0, P_s)

        else:
            for i in reversed(range(self.N_time - 1)):

                m = self.m_update_list[i]
                P = self.P_update_list[i]

                m_pred = self.m_pred_list[i + 1]
                P_pred = self.P_pred_list[i + 1]

                time_int = self.time_int_list[i + 1]
                A = torch.matrix_exp(self.F * time_int)
                # A = self.A_list[i+1]

                G = torch.mm(torch.mm(P, A.T), torch.linalg.pinv(P_pred))
                m_s = m + torch.mm(G, m_s - m_pred)
                P_s = P + torch.mm(torch.mm(G, P_s - P_pred), G.T)

                self.m_smooth_list.insert(0, m_s)
                self.P_smooth_list.insert(0, P_s)


"""
class for dynamic streaming CP model 

different objects/embedding has different time_stamps, assume non-fix-interval by default

new features:
0. no need to know the info of envloved time-stamp list at the beginning - keep update it during the forward  
1. need to store the real-values of time-stamp of all envloved entries
2. need to build a mapping table? (time-stamp -> idx of state) (if we can get this, no longer store 1)
3. during the evaluation, if the time-step in test didn't show in training, merge the neighbor post. (can inplenment in next step)  
"""


class LDS_GP_streaming:

    def __init__(self, hyper_para_dict):

        self.device = hyper_para_dict["device"]  # add the cuda version later

        self.F = hyper_para_dict["F"].double().to(
            self.device)  # transition mat-SDE
        self.H = hyper_para_dict["H"].double().to(
            self.device)  # emission mat-SDE
        self.R = hyper_para_dict["R"].double().to(
            self.device)  # emission noise

        self.P_inf = hyper_para_dict["P_inf"].double().to(self.device)  # P_inf

        self.diffusion_term = hyper_para_dict["diffusion_term"]

        # place-holder
        self.A = None
        self.Q = None



        # self.m_0 = hyper_para_dict["m_0"].double().to(self.device)  # init mean
        self.D = hyper_para_dict["m_0"].shape[0]
        self.m_0 = 1 * torch.randn(self.D, 1).double().to(self.device)
        self.P_0 = hyper_para_dict["P_0"].double().to(self.device)  # init var

        # keep update during the forward pass
        self.current_time_stamp = 0.0
        self.time_int_list = []
        self.time_stamp_list = []
        self.time_2_ind_table = {}

        self.reset_list()

    def reset_list(self):
        self.m = self.m_0  # store the current state(mean)
        self.P = self.P_0  # store the current state(var)

        self.m_update_list = []  # store the filter-update state(mean)
        self.P_update_list = []  # store the filter-update state(var)

        self.m_pred_list = []  # store the filter-pred state(mean)
        self.P_pred_list = []  # store the filter-pred state(var)

        self.m_smooth_list = []  # store the smoothed state(mean)
        self.P_smooth_list = []  # store the smoothed state(mean)

        self.current_time_stamp = 0.0
        self.time_int_list = []
        self.time_stamp_list = []
        self.time_2_ind_table = {}

    def filter_predict(self, time_stamp):

        time_int = time_stamp - self.current_time_stamp

        if len(self.time_int_list)>1 and time_int == self.time_int_list[-1]:

            # reuse the A,Q matrix
            pass
        
        else:
            self.A = torch.matrix_exp(self.F * time_int).double()

            if self.diffusion_term:
                self.Q = self.P_inf - torch.mm(torch.mm(self.A, self.P_inf), self.A.T)
            else:
                self.Q = 0.0

        self.m = torch.mm(self.A, self.m).double()
        self.P = torch.mm(torch.mm(self.A, self.P), self.A.T) + self.Q

        self.m_pred_list.append(self.m)
        self.P_pred_list.append(self.P)

        self.time_int_list.append(time_int)
        self.time_stamp_list.append(time_stamp)
        self.current_time_stamp = time_stamp
        self.time_2_ind_table[time_stamp] = len(self.m_pred_list) - 1

    def filter_update(self, y, R=None, add_to_list=True):

        if R is None:
            R = self.R

        V = y - torch.mm(self.H, self.m)
        S = torch.mm(torch.mm(self.H, self.P), self.H.T) + R
        K = torch.mm(torch.mm(self.P, self.H.T), torch.linalg.pinv(S))
        self.m = self.m + torch.mm(K, V)
        self.P = self.P - torch.mm(torch.mm(K, self.H), self.P)

        if add_to_list:

            self.m_update_list.append(self.m)
            self.P_update_list.append(self.P)

    def smooth(self):

        # start from the last end

        self.N_time = len(self.m_update_list)

        if self.N_time > 0:

            m_s = self.m_update_list[-1]
            P_s = self.P_update_list[-1]

            self.m_smooth_list.insert(0, m_s)
            self.P_smooth_list.insert(0, P_s)

            for i in reversed(range(self.N_time - 1)):

                m = self.m_update_list[i]
                P = self.P_update_list[i]

                m_pred = self.m_pred_list[i + 1]
                P_pred = self.P_pred_list[i + 1]

                time_int = self.time_int_list[i + 1]

                A = torch.matrix_exp(self.F * time_int)

                G = torch.mm(torch.mm(P, A.T), torch.linalg.pinv(P_pred))
                m_s = m + torch.mm(G, m_s - m_pred)
                P_s = P + torch.mm(torch.mm(G, P_s - P_pred), G.T)

                self.m_smooth_list.insert(0, m_s)
                self.P_smooth_list.insert(0, P_s)

    def merge(self):
        pass

    def reset_smooth_list(self):
        self.m_smooth_list = []  # store the smoothed state(mean)
        self.P_smooth_list = []  # store the smoothed state(mean)
