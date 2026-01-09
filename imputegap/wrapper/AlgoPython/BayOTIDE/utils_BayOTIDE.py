import numpy as np
import torch

# import utils
import scipy
from scipy import linalg
import argparse
from torch.utils.data import Dataset
from pathlib import Path
from scipy.special import kn

torch.random.manual_seed(300)


# CRPS score
def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )
def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))

def CRPS_score(target, forecast, eval_points):
    """compute CRPS score"""

    quantiles = np.arange(0.00, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)

    CRPS = 0

    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast, quantiles[i], dim=0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom

    print(CRPS/len(quantiles))
    return CRPS/len(quantiles)


def neg_llk(target,sample_X, test_mask):
    """mean of negative log likelihood of over test points"""
    sample_X_mean =sample_X.mean(dim=0)
    sample_X_std = sample_X.std(dim=0)

    normal_dist = torch.distributions.normal.Normal(sample_X_mean, sample_X_std)

    llk = normal_dist.log_prob(target)

    llk = (llk*test_mask).sum()/test_mask.sum()

    return -llk

def parse_args_dynamic_streaming(argv=None):

    description = "Bayesian dynamic streaming tensor factorization"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=300, help="rand_seed")
    parser.add_argument("--num_fold", type=int, default=1, help="number of folds(random split) and take average,min:1,max:5", )
    parser.add_argument("--machine", type=str, default="zeus", help="machine_name")
    parser.add_argument( '--dataset', type=str, default='guangzhou', help='dataset name: guangzhou, uber or solar')
    parser.add_argument('--task', type=str, default='impute', help='impute or forecast')
    parser.add_argument('--r', type=float, default=0.2, help='train/test ratio, support 0.2 or 0.4 now')
    parser.add_argument('--other_para', type=str, default='', help='')

    args, _unknown = parser.parse_known_args(argv)

    return args


def make_log(args, hyper_dict, result_dict,other_para = ''):

    dict_name = "result_log/" + args.dataset + '_' + args.task + '/'

    kernel_para_trend = hyper_dict['kernel']['kernel_trend']

    kernel_para_season = hyper_dict['kernel']['kernel_season']

    file_name = 'R_' + str(
        hyper_dict['R']
    ) +'_'+ kernel_para_trend['type']+'_'+  kernel_para_season['type']+'_'+ args.machine +other_para+ ".txt"

    Path(dict_name).mkdir(parents=True, exist_ok=True)

    f = open(dict_name + file_name, "a+")

    f.write(
        '\n on %s take %.1f seconds to finish %d folds. avg time: %.1f seconds, \n' %
        (hyper_dict['device'], result_dict['time'], args.num_fold,
         result_dict['time'] / args.num_fold))

    f.write('\n Setting: K_trend = %d , K_season = %d,  n_season = %d, K_bias = %d,  \n'\
    %(
        hyper_dict['K_trend'],hyper_dict['K_season'],
        hyper_dict['n_season'],hyper_dict['K_bias'])
      )

    f.write('\n Kernel-paras: trend-lengthscale = %.2f , trend-variance = %.2f,  season-freq = %s, season-freq  = %s,  \n'\
    %(
    kernel_para_trend['lengthscale'],kernel_para_trend['variance'],
      str(kernel_para_season['freq_list']),
      str(kernel_para_season['lengthscale_list'])))

    f.write('\n CEP_hyper-paras:   Inner_iter=%d ,DAMPING: U = %.1f, W = %.1f, tau = %.1f,  \n'\
    %(
         hyper_dict['INNER_ITER'],
         hyper_dict['DAMPING_U'],
         hyper_dict['DAMPING_W'],hyper_dict['DAMPING_tau']))

    f.write('\n final test RMSE is, avg is %.4f, std is %.4f \n' %
            (result_dict['RMSE'].mean(), result_dict['RMSE'].std()))

    f.write('\n final test MAE is, avg is %.4f, std is %.4f \n' %
            (result_dict['MAE'].mean(), result_dict['MAE'].std()))
    
    f.write('\n final test CRPS is, avg is %.4f, std is %.4f \n' %
            (result_dict['CRPS'].mean(), result_dict['CRPS'].std()))
    
    f.write('\n final test NLLK is, avg is %.4f, std is %.4f \n' %
            (result_dict['NLLK'].mean(), result_dict['NLLK'].std()))
    
    f.write('---------------------------------------------\n')
    f.write('\n\n\n')
    f.close()

    print('log written!')


def make_hyper_dict(config, args=None):
    hyper_dict = config
    hyper_dict["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args is not None:
        # overwrite config with args if needed
        hyper_dict['R'] = args.r
        pass

    return hyper_dict


def make_data_dict(hyper_dict, data_path, fold=0, deep_verbose=False, args=None):
    """to be polish"""
    full_data = np.load(data_path, allow_pickle=True).item()

    if deep_verbose:
        print(f"\t\t{full_data.keys() = }")
        print(f"\t\t{full_data['raw_data'].shape = }\n")
        print(f"\t\t{np.array(full_data['data']).shape = }")
        print(f"\t\t{np.array(full_data['data']) = }\n")
        print(f"\t\t{full_data['time_uni'].shape = }")
        print(f"\t\t{full_data['ndims'] = }")

    # already split into train, test and valid mask
    data_dict = full_data["data"][fold]
    data_dict["data"] = full_data["raw_data"]

    time_scale = hyper_dict["time_scale"]

    data_dict["time_uni"] = full_data["time_uni"] * time_scale

    data_dict["fix_int"] = hyper_dict["fix_int"]

    data_dict["LDS_paras_trend"] = make_LDS_paras_trend(hyper_dict, data_dict)
    
    data_dict["LDS_paras_season"] = make_LDS_paras_season(hyper_dict, data_dict)
    
    if 'clean_data' in full_data.keys():
        data_dict['clean_data'] = full_data['clean_data']
    
    if 'weight' in full_data.keys():
        data_dict['weight'] = full_data['weight']

    # data_dict["LDS_init_list"] = [
    #     make_LDS_paras(dim, hyper_dict, data_dict) for dim in data_dict["ndims"]
    # ]

    return data_dict


def make_LDS_paras_trend(hyper_dict, data_dict):
    LDS_init = {}
    LDS_init["device"] = hyper_dict["device"]

    # build F,H,R
    D = hyper_dict["K_trend"]

    kernel_paras = hyper_dict["kernel"]['kernel_trend']

    LDS_init["diffusion_term"] = True

    LDS_init["R"] = torch.tensor(kernel_paras["noise"])
    if kernel_paras['type'] == "Matern_21":
        LDS_init["F"] = -1 / kernel_paras["lengthscale"] * torch.eye(D)
        LDS_init["H"] = torch.eye(D)
        LDS_init["P_inf"] = torch.eye(D) * kernel_paras["variance"]
        LDS_init["P_0"] = LDS_init["P_inf"]
        LDS_init["m_0"] = torch.randn(D, 1) * 0.3

    elif kernel_paras['type']  == "Matern_23":
        lamb = np.sqrt(3) / kernel_paras["lengthscale"]

        F = torch.zeros((2 * D, 2 * D))
        F[:D, :D] = 0
        F[:D, D:] = torch.eye(D)
        F[D:, :D] = -lamb * lamb * torch.eye(D)
        F[D:, D:] = -2 * lamb * torch.eye(D)

        P_inf = torch.diag(
            torch.cat((
                kernel_paras["variance"] * torch.ones(D),
                lamb * lamb * kernel_paras["variance"] * torch.ones(D),
            )))

        LDS_init["F"] = F
        LDS_init["P_inf"] = P_inf
        LDS_init["H"] = torch.cat((torch.eye(D), torch.zeros(D, D)), dim=1)
        LDS_init["P_0"] = LDS_init["P_inf"]
        LDS_init["m_0"] = 0.1 * torch.ones(2 * D, 1)
    else:

        # rasie not implemented error
        raise NotImplementedError

    return LDS_init


def make_LDS_paras_season(hyper_dict, data_dict):

    LDS_season_list = []
    kernel_paras = hyper_dict["kernel"]['kernel_season']
    for K in range(hyper_dict["K_season"]):

        LDS_init = {}
        LDS_init["device"] = hyper_dict["device"]
        LDS_init["diffusion_term"] = False

        # build F,H,R
        D = hyper_dict["n_season"]

        LDS_init["R"] = torch.tensor(kernel_paras["noise"])

        if kernel_paras['type']  == "exp-periodic":

            freq = kernel_paras["freq_list"][K]
            freq_array = torch.arange(1, D + 1) * freq

            F = torch.zeros((2 * D, 2 * D))
            F[:D, :D] = 0
            F[:D, D:] = torch.diag(-freq_array)

            F[D:, :D] = torch.diag(freq_array)
            F[D:, D:] = 0

            density_freq = density_periodic(D,kernel_paras["lengthscale_list"][K])
                                                
            P_inf = torch.diag(
                torch.cat((
                    density_freq,
                    density_freq,
                )))

            LDS_init["F"] = F
            LDS_init["P_inf"] = P_inf
            LDS_init["H"] = torch.cat((torch.eye(D), torch.zeros(D, D)), dim=1)
            LDS_init["P_0"] = LDS_init["P_inf"]
            LDS_init["m_0"] = 0.1 * torch.ones(2 * D, 1)
        else:

            # rasie not implemented error
            raise NotImplementedError
        
        LDS_season_list.append(LDS_init)

    return LDS_season_list


def density_periodic(n_components, lengthscale, ):
    """density of  exp-periodic kernel, compute using modified Bessel function, see Solin 2014 AISTATS paper for details"""

    result = []

    for order in range(1,n_components+1):
        result.append(2*kn(order, lengthscale**2)/np.exp(lengthscale**2))

    return torch.tensor(np.array(result))



def build_time_data_table(time_ind):
    # input: sorted time-stamp seq (duplicated items exists) attached with data seq
    # output: table (list) of associated data points of each timestamp
    # ref: https://stackoverflow.com/questions/38013778/is-there-any-numpy-group-by-function/43094244
    # attention, here the input "time-stamps" can be either (repeating) id, or exact values, but seq length must match data seq
    # in out table, order of item represents the time id in order
    time_data_table = np.split(
        np.array([i for i in range(len(time_ind))]),
        np.unique(time_ind, return_index=True)[1][1:],
    )
    return time_data_table


def build_id_key_table(nmod, ind):
    # build uid-data_key_table, implement by nested list

    # store the indices of associated nodes in each mode over all obseved entries
    uid_table = []

    # store the indices of obseved entries for each node of each mode
    data_table = []

    if nmod > 1:

        for i in range(nmod):

            values, inv_id = np.unique(ind[:, i], return_inverse=True)

            uid_table.append(list(values))

            sub_data_table = []
            for j in range(len(values)):
                data_id = np.argwhere(inv_id == j)
                if len(data_id) > 1:
                    data_id = data_id.squeeze().tolist()
                else:
                    data_id = [[data_id.squeeze().tolist()]]
                sub_data_table.append(data_id)

            data_table.append(sub_data_table)

    else:
        values, inv_id = np.unique(ind, return_inverse=True)
        uid_table = list(values)
        for j in range(len(values)):
            data_id = np.argwhere(inv_id == j)
            if len(data_id) > 1:
                data_id = data_id.squeeze().tolist()
            else:
                data_id = [[data_id.squeeze().tolist()]]
            data_table.append(data_id)

    return uid_table, data_table


def generate_mask(ndims, ind):
    num_node = sum(ndims)
    nmod = len(ndims)
    ind = torch.tensor(ind)

    mask = torch.zeros((num_node, num_node))
    for i in range(1, nmod):
        row = np.sum(ndims[:i])
        for j in range(i):
            col = np.sum(ndims[:j])
            indij = ind[:, [i, j]]
            indij = torch.unique(indij, dim=0).long()
            row_idx = row + indij[:, 0]
            col_idx = col + indij[:, 1]
            mask[row_idx.long(), col_idx.long()] = 1
    return mask


def generate_Lapla(ndims, ind):
    """
    generate the fixed Laplacian mat of prior K-partition graph,
    which is defined by the observed entries in training set
    """
    num_node = sum(ndims)

    W_init = torch.ones((num_node, num_node))
    mask = generate_mask(ndims, ind)
    Wtril = torch.tril(W_init) * mask
    W = Wtril + Wtril.T
    D = torch.diag(W.sum(1))
    return W - D


def generate_state_space_Matern_23(data_dict, hyper_dict):
    """
    For matern 3/2 kernel with given hyper-paras and data,
    generate the parameters of coorspoding state_space_model,
    recall: for each dim of all-node-embedding, the form of state_space_model is iid (independent & identical)

    input: data_dict, hyper_dict
    output: trans mat: F,  stationary covarianc: P_inf

    """

    ndims = data_dict["ndims"]
    D = sum(ndims)
    ind = data_dict["tr_ind"]

    # hyper-para of kernel
    lengthscale = hyper_dict["ls"]
    variance = hyper_dict["var"]
    c = hyper_dict["c"]  # diffusion rate

    lamb = np.sqrt(3) / lengthscale

    # F = torch.zeros((2*D, 2*D), device=data_dict['device'])
    F = np.zeros((2 * D, 2 * D))
    F[:D, :D] = generate_Lapla(ndims, ind) * c
    F[:D, D:] = np.eye(D)
    F[D:, :D] = -np.square(lamb) * np.eye(D)
    F[D:, D:] = -2 * lamb * np.eye(D)

    Q_c = 4 * lamb**3 * variance * np.eye(D)
    L = np.zeros((2 * D, D))
    L[D:, :] = np.eye(D)
    Q = -np.matmul(np.matmul(L, Q_c), L.T)

    P_inf = Lyapunov_slover(F, Q)

    return torch.tensor(F, device=hyper_dict["device"]), torch.tensor(
        P_inf, device=hyper_dict["device"])


def Lyapunov_slover(F, Q):

    return linalg.solve_continuous_lyapunov(F, Q)


def nan_check_1(model, T):
    msg_list = [
        model.msg_U_llk_m[:, :, T],
        model.msg_U_llk_v[:, :, T],
        model.msg_U_f_m[:, :, T],
        model.msg_U_f_v[:, :, T],
        model.msg_U_b_m[:, :, T],
        model.msg_U_b_v[:, :, T],
        model.msg_U_llk_m_del[:, :, T],
        model.msg_U_llk_v_del[:, :, T],
        model.msg_U_f_m_del[:, :, T],
        model.msg_U_f_v_del[:, :, T],
        model.msg_U_b_m_del[:, :, T],
        model.msg_U_b_v_del[:, :, T],
    ]

    msg_name_list = [
        "msg_U_llk_m",
        "msg_U_llk_v",
        "msg_U_f_m",
        "msg_U_f_v",
        "msg_U_b_m",
        "msg_U_b_v",
        "msg_U_llk_m_del",
        "msg_U_llk_v_del",
        "msg_U_f_m_del",
        "msg_U_f_v_del",
        "msg_U_b_m_del",
        "msg_U_b_v_del",
    ]
    for id, msg in enumerate(msg_list):
        if msg.isnan().any():
            print("invalid number: %s at time %d " % (msg_name_list[id], T))
            return False

    return True


def neg_check_v(model, T):
    msg_list = [
        model.msg_U_llk_v[:, :, T],
        model.msg_U_f_v[:, :, T],
        model.msg_U_b_v[:, :, T],
        model.msg_U_llk_v_del[:, :, T],
        model.msg_U_f_v_del[:, :, T],
        model.msg_U_b_v_del[:, :, T],
    ]

    msg_name_list = [
        "msg_U_llk_v",
        "msg_U_f_v",
        "msg_U_b_v",
        "msg_U_llk_v_del",
        "msg_U_f_v_del",
        "msg_U_b_v_del",
    ]

    for id, msg in enumerate(msg_list):
        if (msg <= 0).any():
            print("invalid v: %s at time %d " % (msg_name_list[id], T))

            return False

    return True


# batch knorker product
def kronecker_product_einsum_batched(A: torch.Tensor, B: torch.Tensor):
    """
    Batched Version of Kronecker Products
    :param A: has shape (b, a, c)
    :param B: has shape (b, k, p)
    :return: (b, ak, cp)
    """
    assert A.dim() == 3 and B.dim() == 3

    res = torch.einsum("bac,bkp->bakcp", A, B).view(A.size(0),
                                                    A.size(1) * B.size(1),
                                                    A.size(2) * B.size(2))
    return res


def Hadamard_product_batch(A: torch.Tensor, B: torch.Tensor):
    """
    Batched Version of Hadamard Products
    :param A: has shape (N, a, b)
    :param B: has shape (N, a, b)
    :return: (N, a, b)
    """
    assert A.dim() == 3 and B.dim() == 3
    assert A.shape == B.shape
    res = A * B
    return res


# batch knorker product
def kronecker_product_einsum_batched(A: torch.Tensor, B: torch.Tensor):
    """
    Batched Version of Kronecker Products
    :param A: has shape (b, a, c)
    :param B: has shape (b, k, p)
    :return: (b, ak, cp)
    """
    assert A.dim() == 3 and B.dim() == 3

    res = torch.einsum("bac,bkp->bakcp", A, B).view(A.size(0),
                                                    A.size(1) * B.size(1),
                                                    A.size(2) * B.size(2))
    return res


def moment_Hadmard(modes,
                   ind,
                   U_m,
                   U_v,
                   order="first",
                   sum_2_scaler=True,
                   device=torch.device("cpu")):
    """
    it's easy to transfer this function to kronecker_product(Tucker form) by changing Hadmard_product_batch to kronecker_product_einsum_batched
    """
    assert order in {"first", "second"}
    assert sum_2_scaler in {True, False}

    last_mode = modes[-1]

    diag_cov = True if U_v[0].size()[-1] == 1 else False

    R_U = U_v[0].size()[1]

    if order == "first":
        # only compute the first order moment

        E_z = U_m[last_mode][ind[:, last_mode]]  # N*R_u*1

        for mode in reversed(modes[:-1]):
            E_u = U_m[mode][ind[:, mode]]  # N*R_u*1
            E_z = Hadamard_product_batch(E_z, E_u)  # N*R_u*1

        return E_z.sum(dim=1) if sum_2_scaler else E_z

    elif order == "second":
        # compute the second order moment E_z / E_z_2

        E_z = U_m[last_mode][ind[:, last_mode]]  # N*R_u*1

        if diag_cov:
            # diagnal cov
            E_z_2 = torch.diag_embed(
                U_v[last_mode][ind[:, last_mode]].squeeze(),
                dim1=1) + torch.bmm(E_z, E_z.transpose(dim0=1,
                                                       dim1=2))  # N*R_u*R_U

        else:
            # full cov
            E_z_2 = U_v[last_mode][ind[:, last_mode]] + torch.bmm(
                E_z, E_z.transpose(dim0=1, dim1=2))  # N*R_u*R_U

        for mode in reversed(modes[:-1]):

            E_u = U_m[mode][ind[:, mode]]  # N*R_u*1

            if diag_cov:

                E_u_2 = torch.diag_embed(
                    U_v[mode][ind[:, mode]].squeeze(), dim1=1) + torch.bmm(
                        E_u, E_u.transpose(dim0=1, dim1=2))  # N*R_u*R_U

            else:
                E_u_2 = U_v[mode][ind[:, mode]] + torch.bmm(
                    E_u, E_u.transpose(dim0=1, dim1=2))  # N*R_u*R_U

            E_z = Hadamard_product_batch(E_z, E_u)  # N*R_u*1
            E_z_2 = Hadamard_product_batch(E_z_2, E_u_2)  # N*R_u*R_u

        if sum_2_scaler:
            E_z = E_z.sum(dim=1)  # N*R_u*1 -> N*1

            # E(1^T z)^2 = trace (1*1^T* z^2)

            E_z_2 = torch.einsum(
                "bii->b",
                torch.matmul(E_z_2,
                             torch.ones(R_U, R_U).to(device))).unsqueeze(
                                 -1)  # N*R_u*R_u -> -> N*1

        return E_z, E_z_2


def moment_Hadmard_T(
        modes,
        ind,
        ind_T,
        U_m_T,
        U_v_T,
        order="first",
        sum_2_scaler=True,
        device=torch.device("cpu"),
):
    assert order in {"first", "second"}
    assert sum_2_scaler in {True, False}

    last_mode = modes[-1]

    diag_cov = True if U_v_T[0].size()[-2] == 1 else False

    R_U = U_v_T[0].size()[1]

    if order == "first":

        E_z = U_m_T[last_mode][ind[:, last_mode], :, :, ind_T]  # N*R_u*1

        for mode in reversed(modes[:-1]):
            E_u = U_m_T[mode][ind[:, mode], :, :, ind_T]  # N*R_u*1
            E_z = Hadamard_product_batch(E_z, E_u)  # N*R_u*1

        return E_z.sum(dim=1) if sum_2_scaler else E_z

    elif order == "second":
        # compute the second order moment E_z / E_z_2

        E_z = U_m_T[last_mode][ind[:, last_mode], :, :, ind_T]  # N*R_u*1

        if diag_cov:
            # diagnal cov
            E_z_2 = torch.diag_embed(
                U_v_T[last_mode][ind[:, last_mode], :, :, ind_T].squeeze(),
                dim1=1) + torch.bmm(E_z, E_z.transpose(dim0=1,
                                                       dim1=2))  # N*R_u*R_U

        else:
            # full cov
            E_z_2 = U_v_T[last_mode][ind[:, last_mode], :, :,
                                     ind_T] + torch.bmm(
                                         E_z, E_z.transpose(
                                             dim0=1, dim1=2))  # N*R_u*R_U

        for mode in reversed(modes[:-1]):

            E_u = U_m_T[mode][ind[:, mode], :, :, ind_T]  # N*R_u*1

            if diag_cov:

                E_u_2 = torch.diag_embed(
                    U_v_T[mode][ind[:, mode], :, :, ind_T].squeeze(),
                    dim1=1) + torch.bmm(E_u, E_u.transpose(
                        dim0=1, dim1=2))  # N*R_u*R_U

            else:
                E_u_2 = U_v_T[mode][ind[:,
                                        last_mode], :, :, ind_T] + torch.bmm(
                                            E_u, E_u.transpose(
                                                dim0=1, dim1=2))  # N*R_u*R_U

            E_z = Hadamard_product_batch(E_z, E_u)  # N*R_u*1
            E_z_2 = Hadamard_product_batch(E_z_2, E_u_2)  # N*R_u*R_u

        if sum_2_scaler:
            E_z = E_z.sum(dim=1)  # N*R_u*1 -> N*1

            # E(1^T z)^2 = trace (1*1^T* z^2)

            E_z_2 = torch.einsum(
                "bii->b",
                torch.matmul(E_z_2,
                             torch.ones(R_U, R_U).to(device))).unsqueeze(
                                 -1)  # N*R_u*R_u -> -> N*1

        return E_z, E_z_2


def aug_time_index(tid, nmod):
    """augmentate batch time-stamp id to tensor-entry-id format
    :paras tid    : list of batch time-stamp id  :shape: N*1
    :paras nmod   : number of modes to augmentate

    :return aug_tid  :shape: N*nmod

    """
    tid_aug = np.stack([tid for i in range(nmod)], axis=1)

    return tid_aug


def get_post(model, T):
    return torch.cat([
        torch.cat([
            model.post_U_m[mode][uid, :, :, T] for uid in model.uid_table[mode]
        ]) for mode in range(model.nmods)
    ])


def moment_product(
        modes,
        ind,
        U_m,
        U_v,
        order="first",
        sum_2_scaler=True,
        device=torch.device("cpu"),
        product_method="hadamard",
):
    """
    it's easy to transfer this function to kronecker_product(Tucker form) by changing Hadmard_product_batch to kronecker_product_einsum_batched

    """
    assert order in {"first", "second"}
    assert sum_2_scaler in {True, False}
    assert product_method in {"hadamard", "kronecker"}

    if product_method == "hadamard":
        product_method = Hadamard_product_batch
    else:
        product_method = kronecker_product_einsum_batched

    last_mode = modes[-1]

    diag_cov = True if U_v[0].size()[-1] == 1 else False

    R_U = U_v[0].size()[1]

    if order == "first":
        # only compute the first order moment

        E_z = U_m[last_mode][ind[:, last_mode]]  # N*R_u*1

        for mode in reversed(modes[:-1]):
            E_u = U_m[mode][ind[:, mode]]  # N*R_u*1
            E_z = product_method(E_z, E_u)  # N*R_u*1

        return E_z.sum(dim=1) if sum_2_scaler else E_z

    elif order == "second":
        # compute the second order moment E_z / E_z_2

        E_z = U_m[last_mode][ind[:, last_mode]]  # N*R_u*1

        if diag_cov:
            # diagnal cov
            E_z_2 = torch.diag_embed(
                U_v[last_mode][ind[:, last_mode]].squeeze(-1),
                dim1=1) + torch.bmm(E_z, E_z.transpose(dim0=1,
                                                       dim1=2))  # N*R_u*R_U

        else:
            # full cov
            E_z_2 = U_v[last_mode][ind[:, last_mode]] + torch.bmm(
                E_z, E_z.transpose(dim0=1, dim1=2))  # N*R_u*R_U

        for mode in reversed(modes[:-1]):

            E_u = U_m[mode][ind[:, mode]]  # N*R_u*1

            if diag_cov:

                E_u_2 = torch.diag_embed(
                    U_v[mode][ind[:, mode]].squeeze(-1), dim1=1) + torch.bmm(
                        E_u, E_u.transpose(dim0=1, dim1=2))  # N*R_u*R_U

            else:
                E_u_2 = U_v[mode][ind[:, mode]] + torch.bmm(
                    E_u, E_u.transpose(dim0=1, dim1=2))  # N*R_u*R_U

            E_z = product_method(E_z, E_u)  # N*R_u*1
            E_z_2 = product_method(E_z_2, E_u_2)  # N*R_u*R_u

        if sum_2_scaler:
            E_z = E_z.sum(dim=1)  # N*R_u*1 -> N*1

            # E(1^T z)^2 = trace (1*1^T* z^2)

            E_z_2 = torch.einsum(
                "bii->b",
                torch.matmul(E_z_2,
                             torch.ones(R_U, R_U).to(device))).unsqueeze(
                                 -1)  # N*R_u*R_u -> -> N*1

        return E_z, E_z_2


def moment_product_T(
        modes,
        ind,
        ind_T,
        U_m_T,
        U_v_T,
        order="first",
        sum_2_scaler=True,
        device=torch.device("cpu"),
        product_method="hadamard",
):

    assert order in {"first", "second"}
    assert sum_2_scaler in {True, False}
    assert product_method in {"hadamard", "kronecker"}

    if product_method == "hadamard":
        product_method = Hadamard_product_batch
    else:
        product_method = kronecker_product_einsum_batched

    last_mode = modes[-1]

    diag_cov = True if U_v_T[0].size()[-1] == 1 else False

    R_U = U_v_T[0].size()[1]

    if order == "first":
        # only compute the first order moment

        E_z = U_m_T[last_mode][ind[:, last_mode], :, :, ind_T]  # N*R_u*1

        for mode in reversed(modes[:-1]):
            E_u = U_m_T[mode][ind[:, mode], :, :, ind_T]  # N*R_u*1
            E_z = product_method(E_z, E_u)  # N*R_u*1

        return E_z.sum(dim=1) if sum_2_scaler else E_z

    elif order == "second":
        # compute the second order moment E_z / E_z_2

        E_z = U_m_T[last_mode][ind[:, last_mode], :, :, ind_T]  # N*R_u*1

        if diag_cov:
            # diagnal cov
            E_z_2 = torch.diag_embed(
                U_v_T[last_mode][ind[:, last_mode], :, :, ind_T].squeeze(-1),
                dim1=1) + torch.bmm(E_z, E_z.transpose(dim0=1,
                                                       dim1=2))  # N*R_u*R_U

        else:
            # full cov
            E_z_2 = U_v_T[last_mode][ind[:, last_mode], :, :,
                                     ind_T] + torch.bmm(
                                         E_z, E_z.transpose(
                                             dim0=1, dim1=2))  # N*R_u*R_U

        for mode in reversed(modes[:-1]):

            E_u = U_m_T[mode][ind[:, mode], :, :, ind_T]  # N*R_u*1

            if diag_cov:

                E_u_2 = torch.diag_embed(
                    U_v_T[mode][ind[:, mode], :, :, ind_T].squeeze(-1),
                    dim1=1) + torch.bmm(E_u, E_u.transpose(
                        dim0=1, dim1=2))  # N*R_u*R_U

            else:
                E_u_2 = U_v_T[mode][ind[:, mode], :, :, ind_T] + torch.bmm(
                    E_u, E_u.transpose(dim0=1, dim1=2))  # N*R_u*R_U

            E_z = product_method(E_z, E_u)  # N*R_u*1
            E_z_2 = product_method(E_z_2, E_u_2)  # N*R_u*R_u

        if sum_2_scaler:
            E_z = E_z.sum(dim=1)  # N*R_u*1 -> N*1

            # E(1^T z)^2 = trace (1*1^T* z^2)

            E_z_2 = torch.einsum(
                "bii->b",
                torch.matmul(E_z_2,
                             torch.ones(R_U, R_U).to(device))).unsqueeze(
                                 -1)  # N*R_u*R_u -> -> N*1

        return E_z, E_z_2
