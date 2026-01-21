"""
Implementation of BayTIDE model, try offline / online version

Similar to previous dynamic tensor factorization model, we use LDS-GP/GP-SS to model the dynamics of each object in each mode

draft link: https://www.overleaf.com/project/6464fdab41aaf0343fd0af11
Author: Shikai Fang
Bellevue,WA, USA, 05/2023
"""

import numpy as np
#from numpy.lib import utils
import torch
from imputegap.wrapper.AlgoPython.BayOTIDE.model_LDS import LDS_GP_streaming
import os
import imputegap.wrapper.AlgoPython.BayOTIDE.utils_BayOTIDE as utils
from torch.distributions.multivariate_normal import MultivariateNormal

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
JITTER = 1e-4

torch.manual_seed(300)


class BayTIDE:

    def __init__(self, hyper_dict, data_dict):
        """-----------------hyper-paras---------------------"""
        self.device = hyper_dict["device"]
        self.K_trend = hyper_dict["K_trend"]  # rank of trend factors
        self.K_season = hyper_dict["K_season"]  # rank of seasonal factors
        self.n_season = hyper_dict["n_season"]  # # of seasonal components


        self.K_bias = hyper_dict["K_bias"]  # rank of bias factors (set as 0 now)
        self.K = self.K_trend + self.K_season*self.n_season + self.K_bias  # total rank

        # prior of noise
        self.v = hyper_dict["v"]  # prior varience of embedding (scaler)
        self.a0 = hyper_dict["a0"]
        self.b0 = hyper_dict["b0"]

        self.DAMPING_U = hyper_dict["DAMPING_U"]
        self.DAMPING_W = hyper_dict["DAMPING_W"]
        self.DAMPING_tau = hyper_dict["DAMPING_tau"]

        # self.nsample_ = hyper_dict["nsample_"]  # # of prediction samples for probabilistic prediction

        """----------------data-dependent paras------------------"""
        # if kernel is matern-1.5, factor = 1, kernel is matern-2.5, factor =2

        self.data = torch.tensor(data_dict["data"]).double().to(self.device)  # N*T

        self.N, self.T = self.data.shape  # N: dim of TS, T: # of time-stamps

        self.mask_train, self.mask_test, self.mask_valid = torch.tensor(data_dict['mask_train']).double().to(self.device), torch.tensor(data_dict['mask_test']).double().to(self.device), torch.tensor(data_dict['mask_valid']).double().to(self.device) # all N*T mask matrix

        self.train_time_ind = np.arange(self.T)  # just set as 1:T at current stage
        self.test_time_ind = np.arange(self.T)  # just set as 1:T at current stage

        # self.unique_train_time = list(np.unique(self.train_time_ind))

        self.time_uni = data_dict['time_uni']  # T * 1, unique time-stamps 
        
        LDS_paras_trend = data_dict["LDS_paras_trend"]  # LDS parameters for trend
        LDS_paras_season_list = data_dict[ 'LDS_paras_season'] # LDS parameters for seasonal (multi-season, save as list)

        # build dynamics (LDS-GP class) for each object in each mode (store using nested list)
        self.traj_class = []

        self.traj_class.append(LDS_GP_streaming(LDS_paras_trend))

        for  LDS_paras_season in  LDS_paras_season_list:
            self.traj_class.append(LDS_GP_streaming(LDS_paras_season))

        # posterior: store the most recently posterior of the traj (temporal factors) from LDS ( column-wise)
        self.post_U_m =   torch.rand( self.K, 1, self.T).double().to(self.device)#  (K, 1, T) 
        self.post_U_v = torch.eye(self.K).reshape( (self.K, self.K, 1)).repeat( 1, 1, self.T).double().to(self.device)  # ( K, K, T) --- recall,  it's a diag matrix

        # self.post_U_v = torch.ones(self.K).reshape( (self.K,1,1)).repeat( 1, 1, self.T).double().to(self.device)  # ( K, 1, T) --diag covariance matrix at each T

        self.post_a = self.a0
        self.post_b = self.b0

        # posterior: store the most recently posterior of the weights (row-wise)
        self.post_W_m =   torch.rand(  self.N, self.K, 1,).double().to(self.device)#  (N,K,1) 
        # self.post_W_m =   torch.ones(  self.N, self.K, 1,).double().to(self.device)#  (N,K,1) 

        self.post_W_v = torch.eye(self.K).reshape( (1, self.K, self.K)).repeat(self.N, 1, 1).double().to(self.device)  # ( N, K, K)


        self.E_tau = torch.tensor([1.0]).to(self.device)  # (1,1,1)



        # build time-data table: Given a time-stamp id, return the indexed of entries
        # self.time_data_table_tr = utils_streaming.build_time_data_table(
        #     self.train_time_ind)

        # self.time_data_table_te = utils_streaming.build_time_data_table(
        #     self.test_time_ind)

        # some place holders

        """message passing related place holders, use  natural parameters of exponential family for efficient computation:
          lam = S_inv, eta = S_inv x m  """
        # store the msg in uid order
        self.msg_U_m = None
        self.msg_U_V = None

        # store the msg in data-llk order
        self.msg_U_lam_llk = None
        self.msg_U_eta_llk = None

        self.msg_a_llk = None
        self.msg_b_llk = None

        self.msg_gamma_lam = None
        self.msg_gamma_eta = None

        # gamma in CP, we just set it as a all-one constant v
        
        self.all_one_vector = torch.ones(self.K,
                                       1).double().to(self.device)  # (R^K)*1

        self.all_one_vector_K =torch.ones(self.K).double().to(self.device)  # (R^K)*1
        
        self.all_one_vector_N = torch.ones(self.N, 1).to(self.device)  # N*1

        self.msg_llk_init_raw()


    def filter_predict(self, T):
        """trajectories of involved objects take KF prediction step + update the posterior"""

        current_time_stamp = self.time_uni[T]

        # filter-predict step for trend factors + update posterior
        trend_factor = self.traj_class[0]

        trend_factor.filter_predict(current_time_stamp)
        H = trend_factor.H
        m = trend_factor.m_pred_list[-1]
        P = trend_factor.P_pred_list[-1]
        self.post_U_m[ :self.K_trend, :, T] = torch.mm(H, m)
        self.post_U_v[ :self.K_trend,:self.K_trend, T] = torch.mm(torch.mm(H, P), H.T)

        current_index = self.K_trend

        # filter-predict step for seasonal factors+ update posterior
        for season_factor in self.traj_class[1:]:
            season_factor.filter_predict(current_time_stamp)
            H = season_factor.H
            m = season_factor.m_pred_list[-1]
            P = season_factor.P_pred_list[-1]
            self.post_U_m[current_index:current_index+self.n_season, :, T] = torch.mm(H, m)
            self.post_U_v[current_index:current_index+self.n_season, current_index:current_index+self.n_season, T] = torch.mm(torch.mm(H, P), H.T)
            current_index += self.n_season

    def filter_update(self,T,add_to_list=True):
        """trajectories take KF update step"""
         # we treat the approx msg of the LLK as the observation values for KF
        y = self.msg_U_m # (K,1)
        R = self.msg_U_V # (K,K)

        # KF update step for each trajectory

        # filter-update step for trend factors + update posterior
        trend_factor = self.traj_class[0]

        trend_factor.filter_update(y=y[:self.K_trend],
                                                     R=R[:self.K_trend,:self.K_trend],
                                                     add_to_list=add_to_list)
        
        # update the posterior
        H = trend_factor.H
        m = trend_factor.m
        P = trend_factor.P
        self.post_U_m[ :self.K_trend, :, T] = torch.mm(H, m)
        self.post_U_v[ :self.K_trend,:self.K_trend, T] = torch.mm(torch.mm(H, P), H.T)

        current_index = self.K_trend

        # filter-update step for seasonal factors+ update posterior
        for season_factor in self.traj_class[1:]:
            season_factor.filter_update(y=y[current_index:current_index+self.n_season],
                                                     R=R[current_index:current_index+self.n_season,current_index:current_index+self.n_season],
                                                     add_to_list=add_to_list)
            H = season_factor.H
            m = season_factor.m
            P = season_factor.P
            self.post_U_m[current_index:current_index+self.n_season, :, T] = torch.mm(H, m)
            self.post_U_v[current_index:current_index+self.n_season, current_index:current_index+self.n_season, T] = torch.mm(torch.mm(H, P), H.T)
            current_index += self.n_season

    def msg_llk_init_raw(self):
        """init the llk-msg used for DAMPING in inner loop of CEP, msg for U and tau and W"""            

        "''init msg for U''"
        self.msg_U_lam_llk_raw =  1e-4 * torch.eye(self.K).reshape((1, self.K, self.K)).repeat(
                self.N, 1, 1).double().to(self.device)  # (N,K,K)

        self.msg_U_eta_llk_raw = 1e-3 * torch.rand(self.N, self.K, 1).double().to(self.device)# (N,K,1)

        "''init msg for W:''"
        self.msg_W_lam_llk_raw =  1e-4 * torch.eye(self.K).reshape((1, self.K, self.K)).repeat(
                self.N, 1, 1).double().to(self.device)  # (N,K,K)

        self.msg_W_eta_llk_raw = 1e-3 * torch.rand(self.N, self.K, 1).double().to(self.device)# (N,K,1)

        "''init msg for tau: TBD''"
        # msg of tau
        self.msg_a_raw = torch.ones(self.N, 1).double().to(self.device)  # N*1
        self.msg_b_raw = torch.ones(self.N, 1).double().to(self.device)  # N*1

    def msg_llk_init(self):
        """init the llk-msg used for DAMPING in inner loop of CEP, msg for U and tau and W"""            

        "''init msg for U''"
        self.msg_U_lam_llk =  self.msg_U_lam_llk_raw # (N,K,K)

        self.msg_U_eta_llk = self.msg_U_eta_llk_raw# (N,K,1)

        "''init msg for W:''"
        self.msg_W_lam_llk =  self.msg_W_eta_llk_raw# (N,K,K)

        self.msg_W_eta_llk = self.msg_W_eta_llk_raw # (N,K,1)

        "''init msg for tau: TBD''"
        # msg of tau
        self.msg_a = self.msg_a_raw  # N*1
        self.msg_b = self.msg_b_raw  # N*1


    def msg_approx_U(self,T):
        """approximate the msg for U(t), as X^{n}(t) = W^{n} x U(t) + noise, we just have:
        msg_U_lam = self.E_tau * E_z_2 
        msg_U_eta = self.y_n * E_z * self.E_tau
        where E_z = mean of W^{n}, E_z_2 = cov of W^{n}+(E_z)^2, n = 1,2,...,N
        """
        E_z = self.post_W_m  # (N,K,1)
        E_z_2 = self.post_W_v + torch.bmm(
                E_z, E_z.transpose(dim0=1, dim1=2))# (N,K,K)

        msg_U_lam_new = self.E_tau * E_z_2  # (N,K,K)
        msg_U_eta_new = self.data[:,T].reshape(-1,1,1) * E_z * self.E_tau   # (N,K,K)

        # DAMPING step:
        self.msg_U_lam_llk = (self.DAMPING_U * self.msg_U_lam_llk +
                                    (1 - self.DAMPING_U) * msg_U_lam_new) # (N,K,K)

        self.msg_U_eta_llk = (self.DAMPING_U * self.msg_U_eta_llk +
                                    (1 - self.DAMPING_U) * msg_U_eta_new) # (N,K,1)
    

        #  apply train-mask! only use the msg from training entries
        mask_train_T = self.mask_train[:,T].reshape(-1,1,1)
        self.msg_U_lam_llk = self.msg_U_lam_llk * mask_train_T # (N,K,K)
        self.msg_U_eta_llk = self.msg_U_eta_llk * mask_train_T  # (N,K,1)

        # filling the msg_U_M, msg_U_V (merge the msg from all dimensions
        msg_V_inv = self.msg_U_lam_llk.sum(dim=0)# (K,K)
        msg_V_inv_m = self.msg_U_eta_llk.sum(dim=0)# (K,1)

        # trans the natural parameters to the mean and cov,
        # these msg will be used for filter update  
        self.msg_U_V =  torch.linalg.inv(msg_V_inv)  # (K,K)
        self.msg_U_m = torch.mm(self.msg_U_V, msg_V_inv_m)  # (K,1)

    def msg_approx_W(self,T):
        """approximate the msg for W, as X^{n}(t) = W^{n} x U(t) + noise, we just have:
        msg_U_lam = self.E_tau * E_z_2 
        msg_U_eta = self.y_n * E_z * self.E_tau
        where E_z = mean of U(t), E_z_2 = cov of U(t)+(E_z)^2, n = 1,2,...,N
        """

        E_z = self.post_U_m[:,:,T].reshape(-1,1)  # (K,1)
        E_z_2 = self.post_U_v[:,:,T] + torch.mm(
                E_z, E_z.T)  # (K,K)
        
        # augument to n-times
        E_z = E_z.reshape(1,self.K,1).repeat(self.N,1,1) # (N,K,1)
        E_z_2 = E_z_2.reshape(1,self.K,self.K).repeat(self.N,1,1) # (N,K,K)
        
        msg_W_lam_new = self.E_tau * E_z_2  # (N,K,K)
        msg_W_eta_new = self.data[:,T].reshape(-1,1,1) * E_z * self.E_tau   # (N,K,1)

        # DAMPING step:
        self.msg_W_lam_llk = (self.DAMPING_W * self.msg_W_lam_llk +
                                    (1 - self.DAMPING_W) * msg_W_lam_new) # (N,K,K)
        
        self.msg_W_eta_llk = (self.DAMPING_W * self.msg_W_eta_llk +
                                    (1 - self.DAMPING_W) * msg_W_eta_new) # (N,K,1)
        
        #  apply train-mask! only use the msg from training entries
        mask_train_T = self.mask_train[:,T].reshape(-1,1,1)
        self.msg_W_lam_llk = self.msg_W_lam_llk * mask_train_T # (N,K,K)
        self.msg_W_eta_llk = self.msg_W_eta_llk * mask_train_T  # (N,K,1)

        # these msg will be used for W_posterior update

    def msg_approx_tau(self, T):
        """approximate the msg for W, as X^{n}(t) = W^{n} x U(t) + noise, we just have:
        noise~Gamma(self.a_tau, self.b_tau)
        msg_a = 1.5 * N
        msg_b = E_{U(t), W} [(X^{n}(t) - W^{n} x U(t) )^2]
        where  n = 1,2,...,N
        """

        E_z_W = self.post_W_m  # (N,K,1)
        E_z_2_W = self.post_W_v + torch.bmm(
                E_z_W, E_z_W.transpose(dim0=1, dim1=2))# (N,K,K)

        E_z_U = self.post_U_m[:,:,T].reshape(-1,1)  # (K,1)
        E_z_2_U = self.post_U_v[:,:,T] + torch.mm(
                E_z_U, E_z_U.T)  # (K,K)
        
        # augument to n-times
        E_z_U = E_z_U.reshape(1,self.K,1).repeat(self.N,1,1) # (N,K,1)
        E_z_2_U = E_z_2_U.reshape(1,self.K,self.K).repeat(self.N,1,1) # (N,K,K)   

        E_z = E_z_W* E_z_U  # (N,K,1)             
        E_z_2 = E_z_2_W * E_z_2_U  # (N,K,K)



        self.msg_a = 1.5 * self.all_one_vector_N  # N*1

        term1 = 0.5 * torch.square(self.data[:,T].reshape(-1,1))  # N*1

        term2 = self.data[:,T].reshape(-1,1) * E_z.sum(dim=1) # N*1

       

        temp = torch.matmul(E_z_2, self.all_one_vector)  # N*K*1
        term3 = 0.5 * torch.matmul(temp.transpose(dim0=1, dim1=2),  self.all_one_vector).reshape(-1, 1)  # N*1

        # alternative way to compute term3, where we have to compute and store E_gamma_2
        # term3 = torch.unsqueeze(0.5* torch.einsum('bii->b',torch.bmm(self.E_gamma_2,self.E_z_2)),dim=-1) # N*1

        self.msg_b = self.DAMPING_tau * self.msg_b + (1 - self.DAMPING_tau) * (
            term1.reshape(-1, 1) - term2.reshape(-1, 1) + term3.reshape(-1, 1)
        )  # N*1

        # mask the msg from training entries
        mask_train_T = self.mask_train[:,T].reshape(-1,1)
        self.msg_a = self.msg_a * mask_train_T
        self.msg_b = self.msg_b * mask_train_T

        # these msg will be used for tau_posterior update

    def post_update_tau(self, T):
        """update post. factor of tau based on current msg. factors"""

        self.post_a = self.post_a + self.msg_a.sum() - self.mask_train[:,T].sum()
        self.post_b = self.post_b + self.msg_b.sum()
        self.E_tau = self.post_a / self.post_b

    def post_update_W(self, T=None):
        """update post. factor of W based on current msg. factors"""

        # transform the current post. W to natural para form
        post_W_lam = torch.linalg.inv(self.post_W_v) # (N,K,K)
        post_W_eta = torch.bmm(post_W_lam,self.post_W_m) # (N,K,1)

        # merge the msg. factors from the llk, than tran back to mean and cov
        self.post_W_v = torch.linalg.inv(post_W_lam + self.msg_W_lam_llk) # (N,K,K)

        self.post_W_m = torch.bmm(self.post_W_v, post_W_eta + self.msg_W_eta_llk) # (N,K,1)




    def smooth(self):
        """smooth back for all objects"""
        for factor in self.traj_class:
            factor.smooth()

    def inner_smooth(self):
        """smooth back for online evaluation during the training, clean out the smooth-result after updating the the post_U"""

        self.smooth()
        self.post_update_U_after_smooth(0)
        for factor in self.traj_class:
            factor.reset_smooth_list()

    def post_update_U_after_smooth(self, T):
        """update post. of U after smoothing, where we extarct the smoothed results from LDS to update the posterior of U along all time stamps"""

        trend_factor = self.traj_class[0]
        H = trend_factor.H
        U_m_smooth = torch.stack(
            [torch.mm(H, m) for m in trend_factor.m_smooth_list], dim=-1)  # (K_trend,1,T)
        
        U_v_smooth = torch.stack(
                    [torch.mm(torch.mm(H, P), H.T) for P in trend_factor.P_smooth_list],
                    dim=-1)  # (K_trend,K_trend,T)
        
        currnt_T = U_m_smooth.shape[-1]
        
        self.post_U_m[:self.K_trend,:,:currnt_T ] = U_m_smooth
        self.post_U_v[:self.K_trend,:self.K_trend,:currnt_T] = U_v_smooth

        current_idx = self.K_trend

        for season_factor in self.traj_class[1:]:

            H = season_factor.H
            U_m_smooth = torch.stack(
            [torch.mm(H, m) for m in season_factor.m_smooth_list], dim=-1)  # (n_season,1,T)

            U_v_smooth = torch.stack(
                        [torch.mm(torch.mm(H, P), H.T) for P in season_factor.P_smooth_list],
                        dim=-1)  # (n_season,n_season,T)

            self.post_U_m[current_idx:current_idx+self.n_season,:,:currnt_T] = U_m_smooth

            self.post_U_v[current_idx:current_idx+self.n_season,current_idx:current_idx+self.n_season,:currnt_T] = U_v_smooth

            current_idx += self.n_season

    def model_test(self, T, prob=False):

        loss_dict = {}
        loss_list = []

        pred = torch.mm(self.post_W_m.squeeze(), self.post_U_m.squeeze())  # (N,T)


        for mask in [self.mask_train, self.mask_valid, self.mask_test]:
            # mask = mask[:,:T]
            # error = (pred[:,:T] - self.data[:,:T]) * mask

            
            error = (pred - self.data) * mask


            RMSE = torch.sqrt(torch.sum(torch.square(error)) / torch.sum(mask))

            MAE = torch.sum(torch.abs(error)) / torch.sum(mask)

            loss_list.append((RMSE.cpu().detach().numpy(),MAE.cpu().detach().numpy()))

        loss_dict['train_RMSE'] = loss_list[0][0]
        loss_dict['train_MAE'] = loss_list[0][1]
        loss_dict['valid_RMSE'] = loss_list[1][0]   
        loss_dict['valid_MAE'] = loss_list[1][1]
        loss_dict['test_RMSE'] = loss_list[2][0]
        loss_dict['test_MAE'] = loss_list[2][1]

        if prob:
            """compute the prob. metric: CRPS and neg-llk"""
            # get the sample of prediction
            sample_X = self.pred_sample(nsample_=100).cpu() # (N,T,nsample)
            neg_llk = utils.neg_llk(self.data.cpu(),sample_X,  self.mask_test.cpu()).cpu().detach().numpy()
            CRPS = utils.CRPS_score(self.data.cpu(),sample_X,  self.mask_test.cpu()).cpu().detach().numpy()

            loss_dict['neg-llk'] = neg_llk  
            loss_dict['CRPS'] = CRPS

        return pred, loss_dict

    def reset(self):
        for factor in self.traj_class:
            factor.reset_list()

    def pred_sample(self,nsample_=100, add_tau = True):

        # generate predicti samples from U, just use mean of W
        with torch.no_grad():
            # generate samples of W from the posterior
            sample_list = []
            for i in range(self.T):
                mean = self.post_U_m[:,:,i].squeeze()
                cov_matrix = torch.diag(self.post_U_v[:,:,i])

                # remove the invalid negative entries in cov-matrix, set as 1e-5
                cov_matrix = torch.where(cov_matrix< 1e-5, self.all_one_vector_K * 1e-5, cov_matrix)
                # cov_matrix = torch.ones(model.K) 

                sample = MultivariateNormal(mean,  torch.diag(cov_matrix.double())).sample(torch.Size([nsample_]))

                sample_list.append(sample)

        sample_U = torch.stack(sample_list,dim=0).permute(1,2,0) # n_sample x K x T

        sample_W = self.post_W_m.reshape(1,self.N,self.K).repeat(nsample_,1,1) # n_sample x N x K

        sample_X = torch.bmm(sample_W,sample_U) # n_sample x N x T

        # add observation noise
        if add_tau:
            sample_X = sample_X + torch.randn_like(sample_X) * torch.sqrt(1/self.E_tau)


        return sample_X


    def filter_update_fake(self,T):
        """no avaliable obseration, do the fake update by just using m and p from predict-step as the update one """

        for factor in self.traj_class:
            factor.m_update_list.append(factor.m_pred_list[-1])
            factor.P_update_list.append(factor.P_pred_list[-1])

        # KF update step for each trajectory

        # filter-update step for trend factors + update posterior
        # trend_factor = self.traj_class[0]

        # trend_factor.filter_update(y=y[:self.K_trend],
        #                                              R=R[:self.K_trend,:self.K_trend],
        #                                              add_to_list=add_to_list)
        
        # # update the posterior
        # H = trend_factor.H
        # m = trend_factor.m
        # P = trend_factor.P
        # self.post_U_m[ :self.K_trend, :, T] = torch.mm(H, m)
        # self.post_U_v[ :self.K_trend,:self.K_trend, T] = torch.mm(torch.mm(H, P), H.T)

        # current_index = self.K_trend

        # # filter-update step for seasonal factors+ update posterior
        # for season_factor in self.traj_class[1:]:
        #     season_factor.filter_update(y=y[current_index:current_index+self.n_season],
        #                                              R=R[current_index:current_index+self.n_season,current_index:current_index+self.n_season],
        #                                              add_to_list=add_to_list)
        #     H = season_factor.H
        #     m = season_factor.m
        #     P = season_factor.P
        #     self.post_U_m[current_index:current_index+self.n_season, :, T] = torch.mm(H, m)
        #     self.post_U_v[current_index:current_index+self.n_season, current_index:current_index+self.n_season, T] = torch.mm(torch.mm(H, P), H.T)
        #     current_index += self.n_season
