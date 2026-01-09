# ===============================================================================================================
# SOURCE: https://github.com/KoheiObata/MissNet/
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://www.dm.sanken.osaka-u.ac.jp/~yasushi/publications/missnet.pdf
# ===============================================================================================================

from argparse import ArgumentParser
import random
import numpy as np
import os
from imputegap.wrapper.AlgoPython.MissNet.main import MissNet
import imputegap.tools.utils as utils_imp


def make_dir(input_dir):
    if not os.path.isdir(input_dir):
        os.makedirs(f"{input_dir}")

class Config:
    def __init__(self, n_components=15, alpha=0.5, beta=1, n_cl=1, max_iter=100, tol=5, random_init=False, verbose=True):
        self.n_components = n_components #latent dimentions
        self.alpha = alpha #trade off the contributions of the contextual matrix and time series
        self.beta = beta #L1-norm
        self.n_cl = n_cl #number of clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_init = random_init
        self.verbose = verbose

class Data:
    def __init__(self, name, max_missing_len=0.05, min_missing_len=0.0, missing_rate_test=0.4, missing_rate_train=0.0, state='test'):
        self.name = name
        self.max_missing_len = max_missing_len
        self.min_missing_len = min_missing_len
        self.missing_rate_test = missing_rate_test
        self.missing_rate_train = missing_rate_train
        self.state = state

def execute(X, SAVEPATH, config):
    #if os.path.isfile(f'{SAVEPATH}/X_impute.txt'):
    #    print(SAVEPATH, 'exists')
    #    return

    #method
    model = MissNet(alpha=config.alpha, beta=config.beta, L=config.n_components, n_cl=config.n_cl)
    history = model.fit(X,
        random_init=config.random_init,
        max_iter=config.max_iter,
        tol=config.tol,
        verbose=config.verbose,
        savedir=SAVEPATH)
    X_impute = model.imputation()

    #save
    make_dir(SAVEPATH)
    model.save_pkl(SAVEPATH)
    #np.savetxt(f'{SAVEPATH}/X_impute.txt', X_impute)

    return X_impute

def data_loop(data_list, config_list):
    if type(data_list.name)!=list:
        data_list.name = [data_list.name]

    for name in data_list.name:
        for max_missing_len in data_list.max_missing_len:
            for min_missing_len in data_list.min_missing_len:
                for missing_rate_test in data_list.missing_rate_test:
                    for missing_rate_train in data_list.missing_rate_train:
                        data = Data(name, max_missing_len=max_missing_len, min_missing_len=min_missing_len, missing_rate_test=missing_rate_test, missing_rate_train=missing_rate_train, state=data_list.state)
                        parameter_loop(data, config_list)

def parameter_loop(data, config_list, iteration=1):
    for n_components in config_list.n_components:
        for alpha in config_list.alpha:
            for beta in config_list.beta:
                for n_cl in config_list.n_cl:
                    for num in range(iteration):
                        random.seed(num)
                        np.random.seed(num)
                        if n_cl==1: tc=0

                        config = Config(n_components=n_components, alpha=alpha, beta=beta, n_cl=n_cl, max_iter=config_list.max_iter, tol=config_list.tol, random_init=config_list.random_init)
                        PATH = f'./data/experiment/{data.name}/MML={data.max_missing_len}_{data.min_missing_len}/MRTe={data.missing_rate_test}/MRTr={data.missing_rate_train}'
                        SAVEPATH = f'./result/{data.name}/MML={data.max_missing_len}_{data.min_missing_len}/MRTe={data.missing_rate_test}/MRTr={data.missing_rate_train}/{data.state}/max_iter={config.max_iter}/tol={config.tol}/n_components={config.n_components}/alpha={config.alpha}/beta={config.beta}/n_cl={config.n_cl}/random_init={config.random_init}/{num}'
                        try:
                            print(SAVEPATH, 'start')
                            X = np.loadtxt(f'{PATH}/X_{data.state}.txt')
                            execute(X, SAVEPATH, config)
                        except:
                            print('Prepare dataset')
                            continue

def experiment_List(site, n_components_list = [30], alpha_list = [0.5], beta_list = [0.1], n_cl_list = [1], max_iter = 50, tol = 5, random_init = False, max_missing_len_list = [0.05], min_missing_len_list = [0.0], missing_rate_test_list = [0.1,0.4,0.8], missing_rate_train_list = [0.0], state = 'test'):
    if site=='dynammo':
        name_list = ['127_07',  '132_32',  '132_43',  '132_46',  '135_07',  '141_04',  '141_16',  '143_22',  '80_10']
        head = f'{site}'

    elif 'synthetic/pattern' in site:
        name_list = []
        for i in range(5):
            name_list.append(f'num={i}')
        head = f'{site}'

    elif site=='synthetic/scale_dim':
        name_list = []
        for d1 in range(20,401,20):
            name_list.append(f'10_{d1}_1000_1')
        head = f'{site}'

    elif site=='synthetic/scale_len':
        name_list = []
        for len in [100,250,500,1000,2500,5000,10000,25000,50000,100000,250000,500000]:
            name_list.append(f'10_50_{len}_1')
        head = f'{site}'

    else:
        name_list = [site]
        head = ''

    for name in name_list:
        name = f'{head}/{name}'
        config_list = Config(n_components=n_components_list, alpha=alpha_list, beta=beta_list, n_cl=n_cl_list, max_iter=max_iter, tol=tol, random_init=random_init)
        data_list = Data(name, max_missing_len_list, min_missing_len_list, missing_rate_test_list, missing_rate_train_list, state)
        data_loop(data_list, config_list)

    return

def experi_id(datasets):
    if datasets=='dynammo':
        name='dynammo'
        experiment_List(name, n_components_list = [30], alpha_list = [0.5], beta_list = [1], n_cl_list = [1], max_iter = 50, tol = 5, random_init = False, max_missing_len_list = [0.05], min_missing_len_list = [0.0], missing_rate_test_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], missing_rate_train_list = [0.1], state = 'test')

    elif datasets=='motes':
        name='motes'
        experiment_List(name, n_components_list = [15], alpha_list = [0.5], beta_list = [1], n_cl_list = [1,2,3], max_iter = 50, tol = 5, random_init = False, max_missing_len_list = [0.05], min_missing_len_list = [0.0], missing_rate_test_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], missing_rate_train_list = [0.1], state = 'test')


    elif datasets=='synthetic/pattern':
        name='synthetic/pattern/10_50_1000_1'
        experiment_List(name, n_components_list = [10], alpha_list = [0.5], beta_list = [1], n_cl_list = [1], max_iter = 50, tol = 5, random_init = False, max_missing_len_list = [0.05], min_missing_len_list = [0.0], missing_rate_test_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],missing_rate_train_list = [0.0], state = 'test')
        name='synthetic/pattern/10_50_1000_2'
        experiment_List(name, n_components_list = [10], alpha_list = [0.5], beta_list = [1], n_cl_list = [2], max_iter = 50, tol = 5, random_init = False, max_missing_len_list = [0.05], min_missing_len_list = [0.0], missing_rate_test_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],missing_rate_train_list = [0.0], state = 'test')

    elif datasets=='synthetic/scale_len':
        name='synthetic/scale_len'
        experiment_List(name, n_components_list = [10], alpha_list = [0.5], beta_list = [1], n_cl_list = [1], max_iter = 2, tol = 1, random_init = False, max_missing_len_list = [0.05], min_missing_len_list = [0.0], missing_rate_test_list = [0.4],missing_rate_train_list = [0.0], state = 'test')

    elif datasets=='synthetic/scale_dim':
        name='synthetic/scale_dim'
        experiment_List(name, n_components_list = [10], alpha_list = [0.5], beta_list = [1], n_cl_list = [1], max_iter = 2, tol = 1, random_init = False, max_missing_len_list = [0.05], min_missing_len_list = [0.0], missing_rate_test_list = [0.4],missing_rate_train_list = [0.0], state = 'test')

    else :
        experiment_List(datasets, n_components_list = [30], alpha_list = [0.5], beta_list = [1], n_cl_list = [1], max_iter = 50, tol = 5, random_init = False, max_missing_len_list = [0.05], min_missing_len_list = [0.0], missing_rate_test_list = [0.1,0.4],missing_rate_train_list = [0.0], state = 'test')




def recovMissNet(X, n_components=15, alpha=0.5, beta=1, n_cl=1, max_iter=100, tol=5, random_init=False, sliding_windows=0, tr_ratio=0.6, verbose=True):

    ts_m = np.copy(X)
    recov = np.copy(X)
    m_mask = np.isnan(X)

    if sliding_windows != 0:
        ts_m = utils_imp.window_truncation(ts_m, seq_len=10, stride=sliding_windows, info="missnet :", verbose=verbose)

    config = Config(n_components=n_components, alpha=alpha, beta=beta, n_cl=n_cl, max_iter=max_iter, tol=tol, random_init=random_init, verbose=verbose)

    if verbose:
        print(f"(IMPUTATION) MissNet2\n\tMatrix Shape: {X.shape[0]}, {X.shape[1]}\n\tn_components: {config.n_components}\n\talpha: {config.alpha}\n\tbeta: {config.beta}\n\tn_cl: {config.n_cl}\n\tmax_iter: {config.max_iter}\n\ttol: {config.tol}\n\trandom_init: {config.random_init}\n\tnumber of nan: {np.count_nonzero(np.isnan(X))}\n\tpercentage: {np.round(np.count_nonzero(np.isnan(X)) / X.size * 100, decimals=1)}%\n\tverbose: {config.verbose}")
        print("\tcall: missnet.impute(params={'n_components':", n_components,", 'alpha':", alpha,", 'beta': ", beta,", 'n_cl':", n_cl,", 'max_iter':", 15,", 'tol': ", tol, ", 'random_init':", random_init,"})\n")

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    dataset_saving_dir = os.path.join(here, "imputegap_assets/models/missnet/")

    imputed_matrix = execute(X=ts_m, SAVEPATH=dataset_saving_dir, config=config)

    if verbose:
        print(f"{imputed_matrix.shape = }")

    if sliding_windows != 0:
        imputed_matrix = utils_imp.reconstruction_windowd_based(preds=imputed_matrix, nbr_timestamps=ts.shape[0], sliding_windows=sliding_windows, verbose=verbose, deep_verbose=False)

    recov[m_mask] = imputed_matrix[m_mask]

    return recov



if __name__ == '__main__':

    parser = ArgumentParser()
    # experiment setting
    #parser.add_argument('--datasets', type=str, default='None', help='dynammo, motes, synthetic/pattern, synthetic/scale_len synthetic/scale_dim')
    args = parser.parse_args()
    args.datasets = "dynammo"
    print(args.__dict__)
    experi_id(args.datasets)
