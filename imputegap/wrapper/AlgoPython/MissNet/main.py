# ===============================================================================================================
# SOURCE: https://github.com/KoheiObata/MissNet/
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://www.dm.sanken.osaka-u.ac.jp/~yasushi/publications/missnet.pdf
# ===============================================================================================================

import warnings
import time
import numpy as np
import pandas as pd
import pickle
import shutil
import os

from imputegap.wrapper.AlgoPython.MissNet.tvgl import TVGL as TimeGraphicalLasso

# Utils
def make_dir(input_dir, delete=False):
    if os.path.isdir(input_dir):
        #print(f'{input_dir} already exist')
        if delete:
            #print('Delete')
            shutil.rmtree(input_dir)
            os.makedirs(input_dir)
            #print(f'{input_dir} is ready')
    else:
        os.makedirs(f"{input_dir}")
        #print(f'{input_dir} is ready')

def interpolate_matrix(X, how='linear'):
    initial_X = pd.DataFrame(X).interpolate(method=how)
    initial_X = initial_X.fillna(method='ffill')
    initial_X = initial_X.fillna(method='bfill')
    return np.array(initial_X)


# Methods
class MissNet:
    def __init__(self, alpha=5e-1, beta=0.1, L=10, n_cl=1):
        #Hyper parameters
        self.alpha = alpha #trade off the contributions of the contextual matrix and time series. If alpha = 0, network is ignored
        self.beta = beta #regularization parameter for sparsity
        self.L = L #hidden dim
        self.n_cl = n_cl #number of clusters

    def initialize(self, X, random_init=False):
        # Given dataset
        self.T = X.shape[0] #time
        self.N = X.shape[1] #dim

        # Initialize model parameters
        self.init_network(X)

        if random_init:
            self.U    = [np.random.rand(self.N, self.L) for _ in range(self.n_cl)] #object latent matrix
            self.B    = np.random.rand(self.L, self.L) #transition matrix
            self.z0   = np.random.rand(self.L) #initial status mean
            self.psi0 = np.random.rand(self.L, self.L) #initial status covariance
            #noise
            self.sgmZ = np.random.rand()
            self.sgmX = [np.random.rand() for _ in range(self.n_cl)]
            self.sgmS = [np.random.rand() for _ in range(self.n_cl)]
            self.sgmV = [np.random.rand() for _ in range(self.n_cl)]
        else:
            self.U    = [np.eye(self.N, self.L) for _ in range(self.n_cl)]
            self.B    = np.eye(self.L)
            self.z0   = np.zeros(self.L)
            self.psi0 = np.eye(self.L)
            #noise
            self.sgmZ = 1.
            self.sgmX = [1. for _ in range(self.n_cl)]
            self.sgmS = [1. for _ in range(self.n_cl)]
            self.sgmV = [1. for _ in range(self.n_cl)]

        # Workspace

        # Forward algorithm
        self.mu_tt = [np.zeros((self.T, self.L)) for _ in range(self.n_cl)] #mu_t|t,i,j: latent status mean at t:i, t-1:j
        self.psi_tt = [np.zeros((self.T, self.L, self.L)) for _ in range(self.n_cl)] #psi_t|t,i,j: latent status covariance at t:i, t-1:j
        self.mu_ = np.zeros((self.T, self.L)) #latent status mean
        self.psi = np.zeros((self.T, self.L, self.L)) #latent status covariance
        self.I   = np.eye(self.L) #identity matrix
        self.P   = np.zeros((self.T, self.L, self.L))

        # Backward algorithm
        self.J    = np.zeros((self.T, self.L, self.L))
        self.zt   = np.zeros((self.T, self.L)) #E[z_t]
        self.ztt  = np.zeros((self.T, self.L, self.L)) #E[z_t,z.T_t]
        self.zt1t = np.zeros((self.T, self.L, self.L)) #E[z_t,z.T_t-1]
        self.mu_h = np.zeros((self.T, self.L))
        self.psih = np.zeros((self.T, self.L, self.L))

        # M-step
        self.v  = [np.zeros((self.N, self.L)) for _ in range(self.n_cl)] #E[v_j]
        self.vv = [np.zeros((self.N, self.L, self.L)) for _ in range(self.n_cl)] #E[v_j,v.T_j]

    def init_network(self, X):
        #initialize cluster assignments
        X_interpolate = interpolate_matrix(X)
        self.F = np.random.choice(range(self.n_cl),self.T)
        self.update_MC()

        #initialize GGM (network)
        self.H  = [np.zeros((self.N, self.N)) for _ in range(self.n_cl)] #covariance of cluster k
        self.G  = [np.zeros(self.N) for _ in range(self.n_cl)] #mean of cluster k
        self.S = [np.zeros((self.N, self.N)) for _ in range(self.n_cl)] #networks
        for k in range(self.n_cl):
            F_k = np.where(self.F==k)[0]
            TVGL = TimeGraphicalLasso(alpha=self.beta, beta=0, max_iter=1000, psi='laplacian', assume_centered=False)
            TVGL.fit(X_interpolate[F_k], np.zeros(X_interpolate[F_k].shape[0]))
            self.H[k] = TVGL.precision_[0]
            self.G[k] = np.nanmean(X_interpolate[F_k], axis=0)
            self.S[k] = self.normalize_precision(TVGL.precision_[0])



    def fit(self, X, random_init=False, max_iter=20, min_iter=3, tol=5, verbose=True, savedir='./temp'):
        """ EM algorithm """

        make_dir(savedir, delete=True)

        W = ~np.isnan(X)

        self.initialize(X, random_init)
        history = {'lle': [], 'time':[]}
        min_lle, tol_counter = np.inf, 0

        for iteration in range(1,max_iter+1):
            tic = time.time()
            try:
                """ E-step """
                # infer F, Z
                lle = self.forward_viterbi(X, W)
                if verbose and self.n_cl>1: print('cluster assignments',[np.count_nonzero(self.F==k) for k in range(self.n_cl)])

                self.backward()
                # infer V
                self.update_latent_context()

                """ M-step """
                # update parameters
                lle -= self.solve_model(X, W, return_loglikelihood=True)

                """ Update the missing values and networks"""
                self.update_networks(X, W) #update G,H,S

                toc = time.time()
                history['time'].append(toc-tic)
                history['lle'].append(lle)
                if verbose:
                    print(f'iter= {iteration}, lle= {lle:.3f}, time= {toc-tic:.3f} [sec]')

                #if LLE no more reduce then convergence (tol=5)
                if iteration > min_iter:
                    if lle < min_lle:
                        min_lle = lle
                        self.save_pkl(savedir) #save the best model
                        tol_counter = 0
                    else:
                        tol_counter += 1

                    if tol_counter >= tol: #Early stopping (end of training)
                        self.load_pkl(savedir) #load the best model
                        return history
            except:
                if verbose: print("EM algorithm Error")
                self.load_pkl(savedir) #load the best model
                return history

        message = "the EM algorithm did not converge, consider increasing 'max_iter'\n"
        if verbose:
            warnings.warn(message)
        self.load_pkl(savedir) #load the best model

        return history


    def forward_viterbi(self, X, W):
        """ Forward algorithm and Viterbi approximation """

        J = np.zeros((self.n_cl, self.T)) #J[i,t]: best LLE at time t state i
        F = np.zeros((self.n_cl, self.T), dtype=int) #J[i,t]: best path at time t state i
        J_tt1 = np.zeros((self.n_cl, self.n_cl, self.T))
        K = [[[[] for _ in range(self.T)] for _ in range(self.n_cl)] for _ in range(self.n_cl)]
        P_ = [np.zeros((self.T, self.L, self.L)) for _ in range(self.n_cl)]
        mu_tt = np.zeros((self.n_cl, self.n_cl, self.T, self.L)) #mu_t|t,i,j: latent status mean at t:i, t-1:j
        psi_tt = np.zeros((self.n_cl, self.n_cl, self.T, self.L, self.L)) #psi_t|t,i,j: latent status covariance at t:i, t-1:j
        mu_tt1 = np.zeros((self.n_cl, self.n_cl, self.T, self.L)) #mu_t|t-1,i,j: latent status mean at t:i, t-1:j
        psi_tt1 = np.zeros((self.n_cl, self.n_cl, self.T, self.L, self.L)) #psi_t|t-1,i,j: latent status covariance at t:i, t-1:j

        for t in range(self.T):
            for i in range(self.n_cl):
                for j in range(self.n_cl):
                    lle = 0
                    ot = W[t] #observed dim
                    xt = X[t, ot] #observed data
                    It = np.eye(xt.shape[0])
                    Ht = self.U[i][ot, :] #observed object latent matrix

                    if t == 0:
                        psi_tt1[i,j,0] = self.psi0
                        mu_tt1[i,j,0] = self.z0
                    else:
                        psi_tt1[i,j,t] = self.B @ self.psi_tt[j][t-1] @ self.B.T + self.sgmZ * self.I #P_t-1 = B @ psi_t-1 @ B.T + sgmZ @ I
                        mu_tt1[i,j,t] = self.B @ self.mu_tt[j][t-1]

                    delta = xt - Ht @ mu_tt1[i,j,t]
                    sigma = Ht @ psi_tt1[i,j,t] @ Ht.T + self.sgmX[i] * It
                    inv_sigma = np.linalg.pinv(sigma)

                    K[i][j][t] = psi_tt1[i,j,t] @ Ht.T @ inv_sigma # K_t = P_t-1 @ H.T_t @ (H_t @ P_t-1 @ H.T_t + sigX @ I).inv
                    mu_tt[i,j,t] = mu_tt1[i,j,t] + K[i][j][t] @ delta
                    psi_tt[i,j,t] = (self.I - K[i][j][t] @ Ht) @ psi_tt1[i,j,t] #psi_t = (I - K_t @ H_t) @ P_t-1

                    #Kalman LLE
                    df = delta @ inv_sigma @ delta / 2
                    sign, logdet = np.linalg.slogdet(inv_sigma)
                    lle -= self.L / 2 * np.log(2 * np.pi)
                    lle += sign * logdet / 2 - df

                    J_tt1[i,j,t] -= lle
                    J_tt1[i,j,t] -= self.window_Gaussian_LLE(X, W, self.U[i], mu_tt[i,j], self.G[i], self.H[i], t)
                    J_tt1[i,j,t] -= np.log(self.A[i,j]) if self.A[i,j]>0 else np.log(0.01)

                if t>0:
                    j_min = np.argmin(J_tt1[i,:,t] + J[:,t-1]) #find j that has minimum cost
                    J[i,t] = J_tt1[i,j_min,t] + J[j_min,t-1] #LLE of j_min
                else:
                    j_min = np.argmin(J_tt1[i,:,t] + 0) #find j that has minimum cost
                    J[i,t] = J_tt1[i,j_min,t] + 0 #LLE of j_min
                F[i,t] = j_min #most likely path
                self.psi_tt[i][t] = psi_tt[i,j_min,t]
                self.mu_tt[i][t] = mu_tt[i,j_min,t]
                P_[i][t] = psi_tt1[i,j_min,t]

        i_min = np.argmin(J[:,self.T-1])
        self.F = F[i_min,:]
        self.F[0] = self.F[1] #first point tends to be different
        self.update_MC()
        for t in range(self.T):
            f_k = self.F[t]
            self.psi[t] = self.psi_tt[f_k][t]
            self.mu_[t] = self.mu_tt[f_k][t]
            self.P[t] = P_[f_k][t]

        return J[i_min][self.T-1]

    def window_Gaussian_LLE(self, X, W, U, mu_tt, mu, invcov, t, window=1):
        if t==0:
            return self.Gaussian_LLE(np.nan_to_num(X[t]) + (1-W[t])*(U@mu_tt[t]), mu, invcov)

        st = 0 if t < window else t-window
        infe = np.array([U@mutt for mutt in mu_tt[st:t]])
        return self.Gaussian_LLE(np.nan_to_num(X[st:t]) + (1-W[st:t])*infe, mu, invcov)

    def Gaussian_LLE(self, x, mu, invcov):
        '''
        input
        x: input data point [d]
        mu: mean [d]
        invcov: presicion matrix [dn, dn]
        '''
        if x.ndim==1:
            N, P = 1, x.shape[0]
            det = np.linalg.det(invcov) if np.linalg.det(invcov)>0 else 1
            lle = -np.log(2*np.pi)*(N*P/2)
            lle += np.log(det)*(N/2) - (np.linalg.multi_dot([x-mu, invcov, (x-mu).T]))/2

        if x.ndim==2:
            N, P = x.shape
            det = np.linalg.det(invcov) if np.linalg.det(invcov)>0 else 1
            lle = -np.log(2*np.pi)*(N*P/2)
            lle += np.log(det)*(N/2) - np.sum(np.diag(np.linalg.multi_dot([x-mu,invcov,(x-mu).T])))/2
        return lle

    def update_MC(self):
        """ Update Markov process """
        # S[i]: counts at state i
        # SS[i,j]: number of transitions from i(t-1) to j(t)
        S = np.zeros((self.n_cl,1))
        SS = np.zeros((self.n_cl,self.n_cl))
        for t in range(1,self.T):
            e = np.zeros((self.n_cl,1))
            e[self.F[t]] = 1

            e1 = np.zeros((self.n_cl,1))
            e1[self.F[t-1]] = 1

            S += e
            SS += e@e1.T
        S = S[:,0]
        SS = SS.T

        # A[i,j]: probability of transitions from j(t-1) to i(t)
        # A[:,j]: sum of the probability of transitions from j(t-1) is 1
        self.A = SS@np.linalg.inv(np.diag(S))
        self.A0 = np.zeros(self.n_cl)
        self.A0[self.F[0]] = 1

    def backward(self):
        """ Backward algorithm """

        self.mu_h[-1] = self.mu_[-1]
        self.psih[-1] = self.psi[-1]

        for t in reversed(range(self.T - 1)):
            self.J[t] = self.psi[t] @ self.B.T @ np.linalg.pinv(self.P[t+1]) #J_t = psi_t @ B.T @ P_t.inv
            self.psih[t] = self.psi[t] + self.J[t] @ (self.psih[t+1] - self.P[t+1]) @ self.J[t].T #psi^_t = psi_t J_t @ (psi^_t+1 - P_t) @ J.T_t
            self.mu_h[t] = self.mu_[t] + self.J[t] @ (self.mu_h[t+1] - self.B @ self.mu_[t]) #mu^_t = mu_t + J_t @ (mu^_t+1 - B @ mu_t)
        self.zt[:] = self.mu_h[:] # E[z_t] = mu^_t

        for t in range(self.T):
            if t > 0:
                self.zt1t[t] = self.psih[t] @ self.J[t-1].T  #E[z_t,z.T_t-1] = psi^_t @ J.T_t-1 + mu^_t @ mu^.T_t-1
                self.zt1t[t] += np.outer(self.mu_h[t], self.mu_h[t-1])
            self.ztt[t] = self.psih[t] + np.outer(self.mu_h[t], self.mu_h[t]) #E[z_t,z.T_t] = psi^_t + mu^_t @ mu^.T_t

    def update_latent_context(self):
        """ Bayes' theorem """
        for k in range(self.n_cl):
            Minv = np.linalg.pinv(self.U[k].T @ self.U[k] + self.sgmS[k] / self.sgmV[k] * np.eye(self.L)) #M = U.T @ U + sgmV.inv @ sgmS @ I
            gamma = self.sgmS[k] * Minv #gamma = sgmS @ ...
            for i in range(self.N):
                self.v[k][i] = Minv @ self.U[k].T @ self.S[k][i, :] #E[v_j] = M.inv @ U.T @ S_j
                self.vv[k][i] = gamma + np.outer(self.v[k][i], self.v[k][i]) #E[v_j,v.T_j] = gamma + E[v_j] @ E[v.T_j]

    def solve_model(self, X, W, return_loglikelihood=False):
        """ Update parameters """

        self.z0 = self.zt[0] #z_0 = E[z_1]
        self.psi0 = self.ztt[0] - np.outer(self.zt[0], self.zt[0]) #psi_0 = E[z_1, z.T_1] - E[z_1]@E[z.T_1]
        self.update_transition_matrix() #update B
        self.update_contextual_covariance() #update sgmV
        self.update_transition_covariance() #update sgmZ
        lle = self.update_object_latent_matrix(X, W, return_loglikelihood) #update U
        self.update_network_covariance() #update sgmS
        self.update_observation_covariance(X, W) #update sgmX

        return lle

    def update_transition_matrix(self):
        self.B = sum(self.zt1t) @ np.linalg.pinv(sum(self.ztt))

    def update_contextual_covariance(self):
        for k in range(self.n_cl):
            val = sum(np.trace(self.vv[k][i]) for i in range(self.N)) / (self.N * self.L)
            self.sgmV[k] = sum(np.trace(self.vv[k][i]) for i in range(self.N)) / (self.N * self.L)

    def update_transition_covariance(self):
        val = np.trace(
            sum(self.ztt[1:])
            - sum(self.zt1t[1:]) @ self.B.T
            - (sum(self.zt1t[1:]) @ self.B.T).T
            + self.B @ sum(self.ztt[:-1]) @ self.B.T
        )
        self.sgmZ = val / ((self.T - 1) * self.L)

    def update_object_latent_matrix(self, X, W, return_loglikelihood=False):
        lle = 0
        for k in range(self.n_cl):
            F_k = np.where(self.F==k)[0]
            F_k = F_k[np.where(F_k!=0)]
            if len(F_k)==0: continue
            for i in range(self.N):
                A1 = self.alpha / self.sgmS[k] * sum(
                    self.S[k][i, j] * self.v[k][j] for j in range(self.N))
                A1 += (1 - self.alpha) / self.sgmX[k] * sum(
                    np.nan_to_num(W[t, i] * X[t, i] * self.zt[t]) for t in F_k) #exclude nan (set nan to 0)
                A2 = self.alpha / self.sgmS[k] * sum(self.vv[k])
                A2 += (1 - self.alpha) / self.sgmX[k] * sum(
                    W[t, i] * self.ztt[t] for t in F_k)
                self.U[k][i, :] = A1 @ np.linalg.pinv(A2)

            if return_loglikelihood:
                for i in range(self.N):
                    # https://www.seas.ucla.edu/~vandenbe/publications/covsel1.pdf
                    delta = self.S[k][i] - self.U[k][i] @ self.v[k][i]
                    sigma = self.sgmS[k] * np.eye(self.N) + self.U[k] @ self.vv[k][i] @ self.U[k].T
                    inv_sigma = np.linalg.pinv(sigma)
                    sign, logdet = np.linalg.slogdet(inv_sigma)
                    lle -= self.L / 2 * np.log(2 * np.pi)
                    lle += sign * logdet / 2 - delta @ inv_sigma @ delta / 2
        return lle

    def update_network_covariance(self):
        for k in range(self.n_cl):
            val = sum(self.S[k][i].T @ self.S[k][i] - 2 * self.S[k][i] @ (self.U[k] @ self.v[k][i]) for i in range(self.N))
            val += np.trace(self.U[k] @ sum(self.vv[k]) @ self.U[k].T)
            self.sgmS[k] = val / self.N ** 2

    def update_observation_covariance(self, X, W):
        for k in range(self.n_cl):
            val = 0
            F_k = np.where(self.F==k)[0]
            F_k = F_k[np.where(F_k!=0)]
            if len(F_k)==0: continue
            for t in F_k:
                ot = W[t, :]
                xt = X[t, ot]
                Ht = self.U[k][ot, :]
                val += np.trace(Ht @ self.ztt[t] @ Ht.T)
                val += xt @ xt - 2 * xt @ (Ht @ self.zt[t])
            self.sgmX[k] = val / W[F_k, :].sum()

    def update_networks(self, X, W):
        """ Impute X and Update network S via graphical lasso """
        for k in range(self.n_cl):
            F_k = np.where(self.F==k)[0]
            if len(F_k)==0: continue
            Y = np.array([self.U[k] @ zt for zt in self.zt])
            X_impute = np.nan_to_num(X) + (1-W)*Y
            TVGL = TimeGraphicalLasso(alpha=self.beta, beta=0, max_iter=1000, psi='laplacian', assume_centered=False)
            TVGL.fit(X_impute[F_k], np.zeros(X_impute[F_k].shape[0]))
            self.H[k] = TVGL.precision_[0]
            self.G[k] = np.nanmean(X_impute[F_k], axis=0)
            self.S[k] = self.normalize_precision(TVGL.precision_[0])

    def normalize_precision(self, H):
        ''' Calculate partial correlation '''
        N, N = H.shape
        S = np.zeros(H.shape)
        for i in range(N):
            for j in range(N):
                if i==j:
                    S[i, j] = 1
                else:
                    S[i, j] = -(H[i, j] / (np.sqrt(H[i, i]) * np.sqrt(H[j, j])))
        return S


    def imputation(self):
        X_impute = np.zeros((self.T, self.N))
        for k in range(self.n_cl):
            F_k = np.where(self.F==k)[0]
            if len(F_k)==0: continue
            Z_temp = np.array([self.U[k] @ zt for zt in self.zt])
            X_impute[F_k] = Z_temp[F_k]
        return X_impute

    def save_pkl(self, outdir):
        with open(f'{outdir}/model.pkl', mode='wb') as f:
            pickle.dump(self, f)

    def load_pkl(self, outdir):
        if os.path.isfile(f'{outdir}/model.pkl'):
            with open(f'{outdir}/model.pkl', mode='rb') as f:
                model = pickle.load(f)

            self.U = model.U
            self.B = model.B
            self.z0 = model.z0
            self.psi0 = model.psi0
            self.sgmZ = model.sgmZ
            self.sgmX = model.sgmX
            self.sgmS = model.sgmS
            self.sgmV = model.sgmV

            self.F = model.F
            self.H = model.H
            self.G = model.G
            self.S = model.S

            # Forward algorithm
            self.mu_tt = model.mu_tt
            self.psi_tt = model.psi_tt
            self.mu_ = model.mu_
            self.psi = model.psi
            self.I = model.I
            self.P = model.P

            # Backward algorithm
            self.J = model.J
            self.zt = model.zt
            self.ztt = model.ztt
            self.zt1t = model.zt1t
            self.mu_h = model.mu_h
            self.psih = model.psih

            # M-step
            self.v = model.v
            self.vv = model.vv