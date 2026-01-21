import ctypes
import os

import numpy as np
import toml
import importlib.resources
import numpy as __numpy_import
import platform


def load_parameters(query: str = "default", algorithm: str = "cdrec", dataset: str = "chlorine", optimizer: str = "b", path=None, verbose=False):
    """
    Load default or optimal parameters for algorithms from a TOML file.

    Parameters
    ----------
    query : str, optional
        'default' or 'optimal' to load default or optimal parameters (default is "default").
    algorithm : str, optional
        Algorithm to load parameters for (default is "cdrec").
    dataset : str, optional
        Name of the dataset (default is "chlorine").
    optimizer : str, optional
    optimizer : str, optional
        Optimizer type for optimal parameters (default is "b").
    path : str, optional
        Custom file path for the TOML file (default is None).
    verbose : bool, optional
        Whether to display the contamination information (default is False).

    Returns
    -------
    tuple
        A tuple containing the loaded parameters for the given algorithm.
    """


    if query == "default":
        if path is None:
            filepath = importlib.resources.files('imputegap.env').joinpath("./default_values.toml")
        else:
            filepath = path
        if not os.path.exists(filepath):
            print(f"\n\tthe selected path is wrong, auto-redirection...")
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            filepath = os.path.join(here, "env/default_values.toml")

    elif query == "optimal":
        algorithm = algorithm.lower().replace("-", "").replace("_", "")
        dataset = dataset.lower().replace("-", "").replace("_", "")
        if path is None:
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            s = "optimal_parameters_" + str(optimizer) + "_" + str(dataset) + "_" + str(algorithm) + ".toml"
            filepath = os.path.join(here, "imputegap_assets/params", s)
        else:
            filepath = path

    else:
        raise ValueError("Query not found for this function (expected 'optimal' or 'default')")

    if not os.path.exists(filepath):
        filepath = "./params/optimal_parameters_" + str(optimizer) + "_" + str(dataset) + "_" + str(algorithm) + ".toml"
        if not os.path.exists(filepath):  # test
            here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            s = "optimal_parameters_" + str(optimizer) + "_" + str(dataset) + "_" + str(algorithm) + ".toml"
            filepath = os.path.join(here, "tests/imputegap_assets/params", s)
        print(f"file not found: {filepath}, load the default folder.\n")

    with open(filepath, "r") as _:
        config = toml.load(filepath)

    if verbose:
        print("\n(SYS) Inner files loaded : ", filepath, "\n")

    if algorithm == "cdrec":
        truncation_rank = int(config[algorithm]['rank'])
        epsilon = float(config[algorithm]['epsilon'])
        iterations = int(config[algorithm]['iteration'])
        return (truncation_rank, epsilon, iterations)
    elif algorithm == "stmvl":
        window_size = int(config[algorithm]['window_size'])
        gamma = float(config[algorithm]['gamma'])
        alpha = int(config[algorithm]['alpha'])
        return (window_size, gamma, alpha)
    elif algorithm == "iim":
        learning_neighbors = int(config[algorithm]['learning_neighbors'])
        if query == "default":
            algo_code = config[algorithm]['algorithm_code']
            return (learning_neighbors, algo_code)
        else:
            return (learning_neighbors,)
    elif algorithm == "mrnn":
        seq_len = int(config[algorithm]['seq_len'])
        epochs = int(config[algorithm]['epochs'])
        batch_size = int(config[algorithm]['batch_size'])
        sliding_windows = int(config[algorithm]['sliding_windows'])
        hidden_layers = int(config[algorithm]['hidden_layers'])
        impute_weight = float(config[algorithm]['impute_weight'])
        num_workers = int(config[algorithm]['num_workers'])
        return (seq_len, epochs, batch_size, sliding_windows, hidden_layers, impute_weight, num_workers)
    elif algorithm == "iterative_svd":
        truncation_rank = int(config[algorithm]['rank'])
        return (truncation_rank)
    elif algorithm == "grouse":
        max_rank = int(config[algorithm]['max_rank'])
        return (max_rank)
    elif algorithm == "dynammo":
        h = int(config[algorithm]['h'])
        max_iteration = int(config[algorithm]['max_iteration'])
        approximation = bool(config[algorithm]['approximation'])
        return (h, max_iteration, approximation)
    elif algorithm == "rosl":
        rank = int(config[algorithm]['rank'])
        regularization = float(config[algorithm]['regularization'])
        return (rank, regularization)
    elif algorithm == "soft_impute":
        max_rank = int(config[algorithm]['max_rank'])
        return (max_rank)
    elif algorithm == "spirit":
        k = int(config[algorithm]['k'])
        w = int(config[algorithm]['w'])
        lvalue = float(config[algorithm]['lvalue'])
        return (k, w, lvalue)
    elif algorithm == "svt":
        tau = float(config[algorithm]['tau'])
        return (tau)
    elif algorithm == "tkcm":
        rank = int(config[algorithm]['rank'])
        return (rank)
    elif algorithm == "deep_mvi":
        max_epoch = int(config[algorithm]['max_epoch'])
        patience = int(config[algorithm]['patience'])
        lr = float(config[algorithm]['lr'])
        batch_size = int(config[algorithm]['batch_size'])
        return (max_epoch, patience, lr, batch_size)
    elif algorithm == "brits":
        model = str(config[algorithm]['model'])
        seq_len = int(config[algorithm]['seq_len'])
        epochs = int(config[algorithm]['epochs'])
        batch_size = int(config[algorithm]['batch_size'])
        sliding_windows = int(config[algorithm]['sliding_windows'])
        hidden_layers = int(config[algorithm]['hidden_layers'])
        impute_weight = float(config[algorithm]['impute_weight'])
        num_workers = int(config[algorithm]['num_workers'])
        return (model, seq_len, epochs, batch_size, sliding_windows, hidden_layers, impute_weight, num_workers)
    elif algorithm == "mpin":
        window = int(config[algorithm]['window'])
        incre_mode = str(config[algorithm]['incre_mode'])
        base = str(config[algorithm]['base'])
        epochs = int(config[algorithm]['epochs'])
        num_of_iteration = int(config[algorithm]['num_of_iteration'])
        k = int(config[algorithm]['k'])
        return (window, incre_mode, base, epochs, num_of_iteration, k)
    elif algorithm == "knn" or algorithm == "knn_impute":
        k = int(config[algorithm]['k'])
        weights = str(config[algorithm]['weights'])
        return (k, weights)
    elif algorithm == "interpolation":
        method = str(config[algorithm]['method'])
        poly_order = int(config[algorithm]['poly_order'])
        return (method, poly_order)
    elif algorithm == "trmf":
        lags = list(config[algorithm]['lags'])
        K = int(config[algorithm]['K'])
        lambda_f = float(config[algorithm]['lambda_f'])
        lambda_x = float(config[algorithm]['lambda_x'])
        lambda_w = float(config[algorithm]['lambda_w'])
        eta = float(config[algorithm]['eta'])
        alpha = float(config[algorithm]['alpha'])
        max_iter = int(config[algorithm]['max_iter'])
        return (lags, K, lambda_f, lambda_x, lambda_w, eta, alpha, max_iter)
    elif algorithm == "mice":
        max_iter = int(config[algorithm]['max_iter'])
        tol = float(config[algorithm]['tol'])
        initial_strategy = str(config[algorithm]['initial_strategy'])
        seed = int(config[algorithm]['seed'])
        return (max_iter, tol, initial_strategy, seed)
    elif algorithm == "miss_forest":
        n_estimators = int(config[algorithm]['n_estimators'])
        max_iter = int(config[algorithm]['max_iter'])
        max_features = str(config[algorithm]['max_features'])
        seed = int(config[algorithm]['seed'])
        return (n_estimators, max_iter, max_features, seed)
    elif algorithm == "xgboost":
        n_estimators = int(config[algorithm]['n_estimators'])
        seed = int(config[algorithm]['seed'])
        return (n_estimators, seed)
    elif algorithm == "miss_net":
        n_components = int(config[algorithm]['n_components'])
        alpha = float(config[algorithm]['alpha'])
        beta = float(config[algorithm]['beta'])
        n_cl = int(config[algorithm]['n_cl'])
        max_iter = int(config[algorithm]['max_iter'])
        tol = float(config[algorithm]['tol'])
        random_init = bool(config[algorithm]['random_init'])
        return (n_components, alpha, beta, n_cl, max_iter, tol, random_init)
    elif algorithm == "gain":
        batch_size = int(config[algorithm]['batch_size'])
        epochs = int(config[algorithm]['epochs'])
        alpha = int(config[algorithm]['alpha'])
        hint_rate = float(config[algorithm]['hint_rate'])
        return (batch_size, epochs, alpha, hint_rate)
    elif algorithm == "grin":
        seq_len = int(config[algorithm]['seq_len'])
        sim_type = str(config[algorithm]['sim_type'])
        epochs = int(config[algorithm]['epochs'])
        batch_size = int(config[algorithm]['batch_size'])
        sliding_windows = int(config[algorithm]['sliding_windows'])
        alpha = int(config[algorithm]['alpha'])
        patience = int(config[algorithm]['patience'])
        num_workers = int(config[algorithm]['num_workers'])
        return (seq_len, sim_type, epochs, batch_size, sliding_windows, alpha, patience, num_workers)
    elif algorithm == "bay_otide":
        K_trend = int(config[algorithm]['K_trend'])
        K_season = int(config[algorithm]['K_season'])
        n_season = int(config[algorithm]['n_season'])
        K_bias = int(config[algorithm]['K_bias'])
        time_scale = int(config[algorithm]['time_scale'])
        a0 = float(config[algorithm]['a0'])
        b0 = float(config[algorithm]['b0'])
        v = float(config[algorithm]['v'])
        num_fold = int(config[algorithm]['num_fold'])
        return (K_trend, K_season, n_season, K_bias, time_scale, a0, b0, v, num_fold)
    elif algorithm == "hkmf_t":
        tags = config[algorithm]['tags']
        seq_len = int(config[algorithm]['seq_len'])
        blackouts_begin = int(config[algorithm]['blackouts_begin'])
        blackouts_end = int(config[algorithm]['blackouts_end'])
        epochs = int(config[algorithm]['epochs'])
        return (tags, seq_len, blackouts_begin, blackouts_end, epochs)
    elif algorithm == "nuwats":
        seq_len = int(config[algorithm]['seq_len'])
        batch_size = int(config[algorithm]['batch_size'])
        epochs = int(config[algorithm]['epochs'])
        gpt_layers = int(config[algorithm]['gpt_layers'])
        num_workers = int(config[algorithm]['num_workers'])
        seed = int(config[algorithm]['seed'])
        return (seq_len, batch_size, epochs,gpt_layers, num_workers, seed)
    elif algorithm == "gpt4ts":
        seq_len = int(config[algorithm]['seq_len'])
        batch_size = int(config[algorithm]['batch_size'])
        epochs = int(config[algorithm]['epochs'])
        gpt_layers = int(config[algorithm]['gpt_layers'])
        num_workers = int(config[algorithm]['num_workers'])
        seed = int(config[algorithm]['seed'])
        return (seq_len, batch_size, epochs, gpt_layers, num_workers, seed)
    elif algorithm == "pristi":
        seq_len = int(config[algorithm]['seq_len'])
        batch_size = int(config[algorithm]['batch_size'])
        epochs = int(config[algorithm]['epochs'])
        sliding_windows = int(config[algorithm]['sliding_windows'])
        target_strategy = str(config[algorithm]['target_strategy'])
        nsamples = int(config[algorithm]['nsamples'])
        num_workers = int(config[algorithm]['num_workers'])
        return (seq_len, batch_size, epochs, sliding_windows, target_strategy, nsamples, num_workers)
    elif algorithm == "csdi":
        seq_len = int(config[algorithm]['seq_len'])
        batch_size = int(config[algorithm]['batch_size'])
        epochs = int(config[algorithm]['epochs'])
        sliding_windows = int(config[algorithm]['sliding_windows'])
        target_strategy = str(config[algorithm]['target_strategy'])
        nsamples = int(config[algorithm]['nsamples'])
        num_workers = int(config[algorithm]['num_workers'])
        return (seq_len, batch_size, epochs, sliding_windows, target_strategy, nsamples, num_workers)
    elif algorithm == "timesnet":
        seq_len = int(config[algorithm]['seq_len'])
        batch_size = int(config[algorithm]['batch_size'])
        epochs = int(config[algorithm]['epochs'])
        gpt_layers = int(config[algorithm]['gpt_layers'])
        num_workers = int(config[algorithm]['num_workers'])
        seed = int(config[algorithm]['seed'])
        return (seq_len, batch_size, epochs, gpt_layers, num_workers, seed)
    elif algorithm == "bit_graph":
        seq_len = int(config[algorithm]['seq_len'])
        sliding_windows = int(config[algorithm]['sliding_windows'])
        kernel_size = int(config[algorithm]['kernel_size'])
        kernel_set = config[algorithm]['kernel_set']
        epochs = int(config[algorithm]['epochs'])
        batch_size = int(config[algorithm]['batch_size'])
        subgraph_size = int(config[algorithm]['subgraph_size'])
        num_workers = int(config[algorithm]['num_workers'])
        return (seq_len, sliding_windows, kernel_size, kernel_set, epochs, batch_size, subgraph_size, num_workers)
    elif algorithm == "saits":
        seq_len = int(config[algorithm]['seq_len'])
        batch_size = int(config[algorithm]['batch_size'])
        epochs = int(config[algorithm]['epochs'])
        sliding_windows = int(config[algorithm]['sliding_windows'])
        n_head = int(config[algorithm]['n_head'])
        num_workers = int(config[algorithm]['num_workers'])
        return (seq_len, batch_size, epochs, sliding_windows, n_head, num_workers)
    elif algorithm == "greedy":
        n_calls = int(config[algorithm]['n_calls'])
        metrics = config[algorithm]['metrics']
        return (n_calls, [metrics])
    elif algorithm.lower() in ["bayesian", "bo", "bayesopt"]:
        n_calls = int(config['bayesian']['n_calls'])
        n_random_starts = int(config['bayesian']['n_random_starts'])
        acq_func = str(config['bayesian']['acq_func'])
        metrics = config['bayesian']['metrics']
        return (n_calls, n_random_starts, acq_func, [metrics])
    elif algorithm.lower() in ['pso', "particle_swarm"]:
        n_particles = int(config['pso']['n_particles'])
        c1 = float(config['pso']['c1'])
        c2 = float(config['pso']['c2'])
        w = float(config['pso']['w'])
        iterations = int(config['pso']['iterations'])
        n_processes = int(config['pso']['n_processes'])
        metrics = config['pso']['metrics']
        return (n_particles, c1, c2, w, iterations, n_processes, [metrics])
    elif algorithm.lower() in  ['sh', "successive_halving"]:
        num_configs = int(config['sh']['num_configs'])
        num_iterations = int(config['sh']['num_iterations'])
        reduction_factor = int(config['sh']['reduction_factor'])
        metrics = config['sh']['metrics']
        return (num_configs, num_iterations, reduction_factor, [metrics])
    elif algorithm.lower() in ['ray_tune', "ray"]:
        metrics = config['ray_tune']['metrics']
        n_calls = int(config['ray_tune']['n_calls'])
        max_concurrent_trials = int(config['ray_tune']['max_concurrent_trials'])
        return ([metrics], n_calls, max_concurrent_trials)
    elif algorithm == "forecaster-naive":
        strategy = str(config[algorithm]['strategy'])
        window_length = int(config[algorithm]['window_length'])
        sp = int(config[algorithm]['sp'])
        return {"strategy": strategy, "window_length": window_length, "sp": sp}
    elif algorithm == "forecaster-exp-smoothing":
        trend = str(config[algorithm]['trend'])
        seasonal = str(config[algorithm]['seasonal'])
        sp = int(config[algorithm]['sp'])
        return {"trend": trend, "seasonal": seasonal, "sp": sp}
    elif algorithm == "forecaster-prophet":
        seasonality_mode = str(config[algorithm]['seasonality_mode'])
        n_changepoints = int(config[algorithm]['n_changepoints'])
        return {"seasonality_mode": seasonality_mode, "n_changepoints": n_changepoints}
    elif algorithm == "forecaster-nbeats":
        input_chunk_length = int(config[algorithm]['input_chunk_length'])
        output_chunk_length = int(config[algorithm]['output_chunk_length'])
        num_blocks = int(config[algorithm]['num_blocks'])
        layer_widths = int(config[algorithm]['layer_widths'])
        random_state = int(config[algorithm]['random_state'])
        n_epochs = int(config[algorithm]['n_epochs'])
        pl_trainer_kwargs = str(config[algorithm]['pl_trainer_kwargs'])
        if pl_trainer_kwargs == "cpu":
            drive = {"accelerator": pl_trainer_kwargs}
        else:
            drive = {"accelerator": pl_trainer_kwargs, "devices": [0]}
        return {"input_chunk_length": input_chunk_length, "output_chunk_length": output_chunk_length, "num_blocks": num_blocks,
                "layer_widths": layer_widths, "random_state": random_state, "n_epochs": n_epochs, "pl_trainer_kwargs": drive}
    elif algorithm == "forecaster-xgboost":
        lags = int(config[algorithm]['lags'])
        return {"lags": lags}
    elif algorithm == "forecaster-lightgbm":
        lags = int(config[algorithm]['lags'])
        verbose = int(config[algorithm]['verbose'])
        return {"lags": lags, "verbose": verbose}
    elif algorithm == "forecaster-lstm":
        input_chunk_length = int(config[algorithm]['input_chunk_length'])
        model = str(config[algorithm]['model'])
        random_state = int(config[algorithm]['random_state'])
        n_epochs = int(config[algorithm]['n_epochs'])
        pl_trainer_kwargs = str(config[algorithm]['pl_trainer_kwargs'])
        if pl_trainer_kwargs == "cpu":
            drive = {"accelerator": pl_trainer_kwargs}
        else:
            drive = {"accelerator": pl_trainer_kwargs, "devices": [0]}
        return {"input_chunk_length": input_chunk_length, "model": model, "random_state": random_state, "n_epochs": n_epochs, "pl_trainer_kwargs": drive}
    elif algorithm == "forecaster-deepar":
        input_chunk_length = int(config[algorithm]['input_chunk_length'])
        model = str(config[algorithm]['model'])
        random_state = int(config[algorithm]['random_state'])
        n_epochs = int(config[algorithm]['n_epochs'])
        pl_trainer_kwargs = str(config[algorithm]['pl_trainer_kwargs'])
        if pl_trainer_kwargs == "cpu":
            drive = {"accelerator": pl_trainer_kwargs}
        else:
            drive = {"accelerator": pl_trainer_kwargs, "devices": [0]}
        return {"input_chunk_length": input_chunk_length, "model": model, "random_state": random_state, "n_epochs": n_epochs, "pl_trainer_kwargs": drive}
    elif algorithm == "forecaster-transformer":
        input_chunk_length = int(config[algorithm]['input_chunk_length'])
        output_chunk_length = int(config[algorithm]['output_chunk_length'])
        random_state = int(config[algorithm]['random_state'])
        n_epochs = int(config[algorithm]['n_epochs'])
        pl_trainer_kwargs = str(config[algorithm]['pl_trainer_kwargs'])
        if pl_trainer_kwargs == "cpu":
            drive = {"accelerator": pl_trainer_kwargs}
        else:
            drive = {"accelerator": pl_trainer_kwargs, "devices": [0]}
        return {"input_chunk_length": input_chunk_length, "output_chunk_length": output_chunk_length, "random_state": random_state, "n_epochs": n_epochs, "pl_trainer_kwargs": drive}

    elif algorithm == "forecaster-hw-add":
        sp = int(config[algorithm]['sp'])
        trend = str(config[algorithm]['trend'])
        seasonal = str(config[algorithm]['seasonal'])
        return {"sp": sp, "trend": trend, "seasonal": seasonal}
    elif algorithm == "forecaster-arima":
        sp = int(config[algorithm]['sp'])
        suppress_warnings = bool(config[algorithm]['suppress_warnings'])
        start_p = int(config[algorithm]['start_p'])
        start_q = int(config[algorithm]['start_q'])
        max_p = int(config[algorithm]['max_p'])
        max_q = int(config[algorithm]['max_q'])
        start_P = int(config[algorithm]['start_P'])
        seasonal = int(config[algorithm]['seasonal'])
        d = int(config[algorithm]['d'])
        D = int(config[algorithm]['D'])
        return {"sp": sp, "suppress_warnings": suppress_warnings, "start_p": start_p, "start_q": start_q,
                "max_p": max_p, "max_q": max_q, "start_P": start_P, "seasonal": seasonal, "d": d, "D": D}
    elif algorithm == "forecaster-sf-arima":
        sp = int(config[algorithm]['sp'])
        start_p = int(config[algorithm]['start_p'])
        start_q = int(config[algorithm]['start_q'])
        max_p = int(config[algorithm]['max_p'])
        max_q = int(config[algorithm]['max_q'])
        start_P = int(config[algorithm]['start_P'])
        seasonal = int(config[algorithm]['seasonal'])
        d = int(config[algorithm]['d'])
        D = int(config[algorithm]['D'])
        return {"sp": sp, "start_p": start_p, "start_q": start_q,
                "max_p": max_p, "max_q": max_q, "start_P": start_P, "seasonal": seasonal, "d": d, "D": D}
    elif algorithm == "forecaster-bats":
        sp = int(config[algorithm]['sp'])
        use_trend = bool(config[algorithm]['use_trend'])
        use_box_cox = bool(config[algorithm]['use_box_cox'])
        return {"sp": sp, "use_trend": use_trend, "use_box_cox": use_box_cox}
    elif algorithm == "forecaster-ets":
        sp = int(config[algorithm]['sp'])
        auto = bool(config[algorithm]['auto'])
        return {"sp": sp, "auto": auto}
    elif algorithm == "forecaster-croston":
        smoothing = float(config[algorithm]['smoothing'])
        return {"smoothing": smoothing}
    elif algorithm == "forecaster-unobs":
        level = bool(config[algorithm]['level'])
        trend = bool(config[algorithm]['trend'])
        sp = int(config[algorithm]['sp'])
        return {"level": level, "trend": trend, "seasonal": sp}
    elif algorithm == "forecaster-theta":
        sp = int(config[algorithm]['sp'])
        deseasonalize = bool(config[algorithm]['deseasonalize'])
        return {"sp": sp, "deseasonalize": deseasonalize}
    elif algorithm == "forecaster-rnn":
        input_size = int(config[algorithm]['input_size'])
        inference_input_size = int(config[algorithm]['inference_input_size'])
        return {"input_size": input_size, "inference_input_size": inference_input_size}
    elif algorithm == "colors":
        colors = config[algorithm]['plot']
        return colors
    elif algorithm == "colors_blacks":
        colors = config[algorithm]['plot']
        return colors
    elif algorithm == "other":
        return config

    # Your own default parameters #contributing
    #
    #elif algorithm == "your_algo_name":
    #    param_1 = int(config[algorithm]['param_1'])
    #    param_2 = config[algorithm]['param_2']
    #    param_3 = float(config[algorithm]['param_3'])
    #    return (param_1, param_2, param_3)

    else:
        print("(SYS) Default/Optimal config not found for this algorithm")
        return None


def config_impute_algorithm(incomp_data, algorithm, verbose=True):
    """
    Configure and execute algorithm for selected imputation imputer and pattern.

    Parameters
    ----------
    incomp_data : TimeSeries
        TimeSeries object containing dataset.

    algorithm : str
        Name of algorithm

    verbose : bool, optional
        Whether to display the contamination information (default is False).

    Returns
    -------
    BaseImputer
        Configured imputer instance with optimal parameters.
    """

    from imputegap.recovery.imputation import Imputation
    from imputegap.recovery.manager import TimeSeries

    alg_low = algorithm.lower()
    alg = alg_low.replace('_', '').replace('-', '')

    # 1st generation
    if alg == "cdrec":
        imputer = Imputation.MatrixCompletion.CDRec(incomp_data)
    elif alg == "stmvl":
        imputer = Imputation.PatternSearch.STMVL(incomp_data)
    elif alg == "iim":
        imputer = Imputation.MachineLearning.IIM(incomp_data)
    elif alg == "mrnn":
        imputer = Imputation.DeepLearning.MRNN(incomp_data)

    # 2nd generation
    elif alg == "iterativesvd" or alg == "itersvd":
        imputer = Imputation.MatrixCompletion.IterativeSVD(incomp_data)
    elif alg == "grouse":
        imputer = Imputation.MatrixCompletion.GROUSE(incomp_data)
    elif alg == "dynammo":
        imputer = Imputation.PatternSearch.DynaMMo(incomp_data)
    elif alg == "rosl":
        imputer = Imputation.MatrixCompletion.ROSL(incomp_data)
    elif alg == "softimpute" or alg == "softimp":
        imputer = Imputation.MatrixCompletion.SoftImpute(incomp_data)
    elif alg == "spirit":
        imputer = Imputation.MatrixCompletion.SPIRIT(incomp_data)
    elif alg == "svt":
        imputer = Imputation.MatrixCompletion.SVT(incomp_data)
    elif alg == "tkcm":
        imputer = Imputation.PatternSearch.TKCM(incomp_data)
    elif alg == "deepmvi":
        imputer = Imputation.DeepLearning.DeepMVI(incomp_data)
    elif alg == "brits":
        imputer = Imputation.DeepLearning.BRITS(incomp_data)
    elif alg == "mpin":
        imputer = Imputation.DeepLearning.MPIN(incomp_data)
    elif alg == "pristi":
        imputer = Imputation.DeepLearning.PriSTI(incomp_data)

    # 3rd generation
    elif alg == "knn" or alg == "knnimpute":
        imputer = Imputation.Statistics.KNNImpute(incomp_data)
    elif alg == "interpolation":
        imputer = Imputation.Statistics.Interpolation(incomp_data)
    elif alg == "meanseries" or alg == "meanimputebyseries":
        imputer = Imputation.Statistics.MeanImputeBySeries(incomp_data)
    elif alg == "minimpute":
        imputer = Imputation.Statistics.MinImpute(incomp_data)
    elif alg == "zeroimpute":
        imputer = Imputation.Statistics.ZeroImpute(incomp_data)
    elif alg == "trmf":
        imputer = Imputation.MatrixCompletion.TRMF(incomp_data)
    elif alg == "mice":
        imputer = Imputation.MachineLearning.MICE(incomp_data)
    elif alg == "missforest":
        imputer = Imputation.MachineLearning.MissForest(incomp_data)
    elif alg == "xgboost":
        imputer = Imputation.MachineLearning.XGBOOST(incomp_data)
    elif alg == "missnet":
        imputer = Imputation.DeepLearning.MissNet(incomp_data)
    elif alg == "gain":
        imputer = Imputation.DeepLearning.GAIN(incomp_data)
    elif alg == "grin":
        imputer = Imputation.DeepLearning.GRIN(incomp_data)
    elif alg == "bayotide":
        imputer = Imputation.DeepLearning.BayOTIDE(incomp_data)
    elif alg == "hkmft" or alg == "hkmf-t" :
        imputer = Imputation.DeepLearning.HKMFT(incomp_data)
    elif alg == "bitgraph":
        imputer = Imputation.DeepLearning.BitGraph(incomp_data)
    elif alg == "meanimpute":
        imputer = Imputation.Statistics.MeanImpute(incomp_data)

    # 4th generation
    elif alg == "nuwats":
        imputer = Imputation.LLMs.NuwaTS(incomp_data)
    elif alg == "gpt4ts":
        imputer = Imputation.LLMs.GPT4TS(incomp_data)

    # 5th generation
    elif alg == "saits":
        imputer = Imputation.DeepLearning.SAITS(incomp_data)
    elif alg == "timesnet":
        imputer = Imputation.DeepLearning.TimesNet(incomp_data)
    elif alg == "csdi":
        imputer = Imputation.DeepLearning.CSDI(incomp_data)

    # your own implementation #contributing
    #
    #elif alg == "your_algo_name":
    #    imputer = Imputation.MyFamily.NewAlg(incomp_data)
    else:
        raise ValueError(f"(IMP) Algorithm '{algorithm}' not recognized, please choose your algorithm from this list:\n\t{TimeSeries().algorithms}")
        imputer = None

    if imputer is not None:
        imputer.verbose = verbose

    return imputer


def save_optimization(optimal_params, algorithm="cdrec", dataset="", optimizer="b", file_name=None, verbose=True):
    """
    Save the optimization parameters to a TOML file for later use without recomputing.

    Parameters
    ----------
    optimal_params : dict
        Dictionary of the optimal parameters.

    algorithm : str, optional
        The name of the imputation algorithm (default is 'cdrec').

    dataset : str, optional
        The name of the dataset (default is an empty string).

    optimizer : str, optional
        The name of the optimizer used (default is 'b').

    file_name : str, optional
        The name of the TOML file to save the results (default is None).

    Returns
    -------
    None
    """

    algorithm = algorithm.lower().replace("-", "").replace("_", "")
    dataset = dataset.lower().replace("-", "").replace("_", "")

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    s = "optimal_parameters_" + str(optimizer) + "_" + str(dataset) + "_" + str(algorithm) + ".toml"
    dir_name = os.path.join(here, "imputegap_assets/params")
    file_name = os.path.join(here, "imputegap_assets/params", s)

    if isinstance(optimal_params, dict):
        optimal_params = tuple(optimal_params.values())

    dir_name = os.path.dirname(dir_name)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)


    if algorithm == "cdrec":
        params_to_save = {
            "rank": int(optimal_params[0]),
            "epsilon": optimal_params[1],
            "iteration": int(optimal_params[2])
    }
    elif algorithm == "mrnn" or algorithm == "MRNN":
        params_to_save = {
            "seq_len": int(optimal_params[0]),
            "epoch": int(optimal_params[1]),
            "batch_size": int(optimal_params[2]),
            "sliding_windows": int(optimal_params[3]),
            "hidden_layers": int(optimal_params[4]),
            "impute_weight": float(optimal_params[5]),
            "num_workers": int(optimal_params[6])
        }
    elif algorithm == "stmvl" or algorithm == "STMVL" or algorithm == "ST-MVL":
        params_to_save = {
            "window_size": int(optimal_params[0]),
            "gamma": optimal_params[1],
            "alpha": int(optimal_params[2])
        }
    elif algorithm == "iim" or algorithm == "IIM":
        params_to_save = {
            "learning_neighbors": int(optimal_params[0])
        }

    elif algorithm == "iterative_svd" or algorithm == "iterativesvd":
        params_to_save = {
            "rank": int(optimal_params[0])
        }
    elif algorithm == "grouse" or algorithm == "GROUSE":
        params_to_save= {
            "max_rank": int(optimal_params[0])
        }
    elif algorithm == "rosl" or algorithm == "ROSL":
        params_to_save = {
            "rank": int(optimal_params[0]),
            "regularization": optimal_params[1]
        }
    elif algorithm == "soft_impute" or algorithm == "softimpute":
        params_to_save = {
            "max_rank": int(optimal_params[0])
        }
    elif algorithm == "spirit" or algorithm == "SPIRIT":
        params_to_save = {
            "k": int(optimal_params[0]),
            "w": int(optimal_params[1]),
            "lvalue": optimal_params[2]
        }
    elif algorithm == "svt" or algorithm == "SVT":
        params_to_save = {
            "tau": optimal_params[0],
            "delta": optimal_params[1],
            "max_iter": int(optimal_params[2])
        }
    elif algorithm == "dynammo" or algorithm == "DynamoMO":
        params_to_save = {
            "h": int(optimal_params[0]),
            "max_iteration": int(optimal_params[1]),
            "approximation": bool(optimal_params[2])
        }
    elif algorithm == "tkcm":
        params_to_save = {
            "rank": int(optimal_params[0])
        }
    elif algorithm == "brits":
        params_to_save = {
            "model": optimal_params[0],
            "seq_len": int(optimal_params[1]),
            "epoch": int(optimal_params[2]),
            "batch_size": int(optimal_params[3]),
            "sliding_windows": int(optimal_params[4]),
            "hidden_layers": int(optimal_params[5]),
            "impute_weight": float(optimal_params[6]),
            "num_workers": int(optimal_params[7])
        }
    elif algorithm == "deep_mvi" or algorithm == "deepmvi":
        params_to_save = {
            "max_epoch": int(optimal_params[0]),
            "patience": int(optimal_params[1]),
            "lr": float(optimal_params[2]),
            "batch_size": float(optimal_params[3])
        }
    elif algorithm == "mpin":
        params_to_save = {
            "window": int(optimal_params[0]),
            "incre_mode": optimal_params[1],
            "base": optimal_params[2],
            "epochs": int(optimal_params[3]),
            "num_of_iteration": int(optimal_params[4]),
            "k": int(optimal_params[5])
        }
    elif algorithm == "knn" or algorithm == "knn_impute":
        params_to_save = {
            "k": int(optimal_params[0]),
            "weights": str(optimal_params[1])
        }
    elif algorithm == "interpolation":
        params_to_save = {
            "method": str(optimal_params[0]),
            "poly_order": int(optimal_params[1])
        }
    elif algorithm == "mice":
        params_to_save = {
            "max_iter": int(optimal_params[0]),
            "tol": float(optimal_params[1]),
            "initial_strategy": str(optimal_params[2]),
            "seed": 42
        }
    elif algorithm == "miss_forest" or algorithm == "missforest":
        params_to_save = {
            "n_estimators": int(optimal_params[0]),
            "max_iter": int(optimal_params[1]),
            "max_features": str(optimal_params[2]),
            "seed": 42
        }
    elif algorithm == "xgboost":
        params_to_save = {
            "n_estimators": int(optimal_params[0]),
            "seed": 42
        }
    elif algorithm == "miss_net" or algorithm == "missnet":
        params_to_save = {
            "n_components": int(optimal_params[0]),
            "alpha": float(optimal_params[1]),
            "beta": float(optimal_params[2]),
            "n_cl": int(optimal_params[3]),
            "max_iter": int(optimal_params[4]),
            "tol": float(optimal_params[5]),
            "random_init": bool(optimal_params[6])
    }
    elif algorithm == "gain":
        params_to_save = {
            "batch_size": int(optimal_params[0]),
            "epochs": int(optimal_params[1]),
            "alpha": int(optimal_params[2]),
            "hint_rate": float(optimal_params[3]),
        }
    elif algorithm == "grin":
        params_to_save = {
            "seq_len": int(optimal_params[0]),
            "sim_type": str(optimal_params[1]),
            "epochs": int(optimal_params[2]),
            "batch_size": int(optimal_params[3]),
            "sliding_windows": int(optimal_params[4]),
            "alpha": int(optimal_params[5]),
            "patience": int(optimal_params[6]),
            "num_workers": int(optimal_params[7])
        }
    elif algorithm == "bay_otide" or algorithm == "bayotide":
        params_to_save = {
            "K_trend": int(optimal_params[0]),
            "K_season": int(optimal_params[1]),
            "n_season": int(optimal_params[2]),
            "K_bias": int(optimal_params[3]),
            "time_scale": int(optimal_params[4]),
            "a0": float(optimal_params[5]),
            "b0": float(optimal_params[6]),
            "v": float(optimal_params[7]),
            "num_fold": int(optimal_params[8]),
        }
    elif algorithm == "hkmf_t" or algorithm == "hkmft":
        params_to_save = {
            "tags": optimal_params[0],
            "seq_len": optimal_params[1],
            "blackouts_begin": int(optimal_params[2]),
            "blackouts_end": int(optimal_params[3]),
            "epochs": int(optimal_params[4]),
        }
    elif algorithm == "bit_graph" or algorithm == "bitgraph":
        params_to_save = {
            "seq_len": int(optimal_params[0]),
            "sliding_windows": int(optimal_params[1]),
            "kernel_size": int(optimal_params[2]),
            "kernel_set": optimal_params[3],
            "epochs": int(optimal_params[4]),
            "batch_size": int(optimal_params[5]),
            "subgraph_size": int(optimal_params[6]),
            "num_workers": int(optimal_params[7])
        }
    elif algorithm == "nuwats" or algorithm == "NUWATS":
        params_to_save = {
            "seq_len": int(optimal_params[0]),
            "batch_size": float(optimal_params[1]),
            "epochs": int(optimal_params[2]),
            "gpt_layers": int(optimal_params[3]),
            "num_workers": int(optimal_params[4]),
            "seed": int(optimal_params[5]),
        }
    elif algorithm == "gpt4ts" or algorithm == "GPT4TS":
        params_to_save = {
            "seq_len": int(optimal_params[0]),
            "batch_size": float(optimal_params[1]),
            "epochs": int(optimal_params[2]),
            "gpt_layers": int(optimal_params[3]),
            "num_workers": int(optimal_params[4]),
            "seed": int(optimal_params[5]),
        }
    elif algorithm == "timesnet" or algorithm == "TimesNet":
        params_to_save = {
            "seq_len": int(optimal_params[0]),
            "batch_size": float(optimal_params[1]),
            "epochs": int(optimal_params[2]),
            "gpt_layers": int(optimal_params[3]),
            "num_workers": int(optimal_params[4]),
            "seed": int(optimal_params[5]),
        }
    elif algorithm == "pristi":
        params_to_save = {
            "seq_len": int(optimal_params[0]),
            "batch_size": float(optimal_params[1]),
            "epochs": int(optimal_params[2]),
            "sliding_windows": int(optimal_params[3]),
            "target_strategy": str(optimal_params[4]),
            "nsamples": int(optimal_params[5]),
            "num_workers": int(optimal_params[6]),
        }
    elif algorithm == "csdi" or algorithm == "CSDI":
        params_to_save = {
            "seq_len": int(optimal_params[0]),
            "batch_size": float(optimal_params[1]),
            "epochs": int(optimal_params[2]),
            "sliding_windows": int(optimal_params[3]),
            "target_strategy": str(optimal_params[4]),
            "nsamples": int(optimal_params[5]),
            "num_workers": int(optimal_params[6]),
        }
    elif algorithm == "saits" or algorithm == "SAITS":
        params_to_save = {
            "seq_len": int(optimal_params[0]),
            "batch_size": int(optimal_params[1]),
            "epochs": int(optimal_params[2]),
            "sliding_windows": int(optimal_params[3]),
            "n_head": int(optimal_params[4]),
            "num_workers": int(optimal_params[5])
        }

    # Your own optimal save parameters #contributing
    #
    #elif algorithm == "your_algo_name":
    #    params_to_save = {
    #        "param_1": int(optimal_params[0]),
    #    "param_2": optimal_params[1],
    #    "param_3": float(optimal_params[2]),
    #}

    else:
        if verbose:
            print(f"\n\t\t(SYS) Algorithm {algorithm} is not recognized.")
        return

    toml_payload = {algorithm: params_to_save}

    try:
        with open(file_name, 'w') as file:
            toml.dump(toml_payload, file)
        if verbose:
            print(f"\n(SYS) Optimization parameters successfully saved to {file_name}")
    except Exception as e:
        print(f"\n(SYS) An error occurred while saving the file: {e}")

    return file_name


def check_family(family="DeepLearning", algorithm=""):
    """
    Check whether a given algorithm belongs to a specified family.

    Parameters
    ----------
    family : str, optional
        Name of the algorithm family to check against (e.g. ``"DeepLearning"``).
        Defaults to ``"DeepLearning"``.
    algorithm : str
        Name of the algorithm to check for membership in the given family.
        Matching is case-insensitive and ignores underscores and hyphens.

    Returns
    -------
    bool
        ``True`` if an algorithm with the given name exists within the
        specified family, ``False`` otherwise.
    """
    norm_input = algorithm.lower().replace("_", "").replace("-", "")

    for full_name in list_of_algorithms_with_families():
        if full_name.startswith(family+"."):
            suffix = full_name.split(".", 1)[1]
            norm_suffix = suffix.lower().replace("_", "").replace("-", "")

            if norm_input == norm_suffix:
                return True
    return False

def sets_splitter_based_on_training(tr, split=0.66667, verbose=False):
    """
    Compute test and validation split ratios based on a given training ratio.

    Ensures that the sum of training, validation, and test ratios equals 1.0
    after rounding to one decimal place. Raises a ValueError if the resulting
    ratios do not sum to 1.0 within tolerance.

    Parameters
    ----------
    tr : float
        Training ratio (between 0 and 1).

    split : float, optional
         Percentage of test set. Default is 2/3.

    verbose : bool, optional
        If True, prints the computed ratios for verification. Default is False.

    Returns
    -------
    - test_ratio : Fraction of data allocated to the test set.
    - val_ratio : Fraction of data allocated to the validation set.

    Raises
    ------
    ValueError
        If the computed ratios do not sum to 1.0 (after rounding).
    """
    test_len = round((1 - tr) * (split), 1)
    val_len = round(1 - tr - test_len, 1)

    total = round(tr + test_len + val_len, 1)
    if total != 1.0 or test_len < 0 or val_len < 0:
        raise ValueError(f"Ratios do not sum to 1.0 (train={tr}, test={test_len}, val={val_len}, total={total})")

    if verbose:
        print(f"\ntraining ratio: {tr}, validation ratio: {val_len}, test ratio: {test_len}")
        print(f"\tsum of ratios: {total}\n")

    return test_len, val_len


def config_contamination(ts, pattern, dataset_rate=0.4, series_rate=0.4, block_size=10, offset=0.1, seed=True, limit=1, shift=0.05, std_dev=0.5, explainer=False, probabilities=None, logic_by_series=True, verbose=True):
    """
    Configure and execute contamination for selected imputation algorithm and pattern.

    Parameters
    ----------
    rate : float
        Mean parameter for contamination missing percentage rate.
    ts_test : TimeSeries
        A TimeSeries object containing dataset.
    pattern : str
        Type of contamination pattern (e.g., "mcar", "mp", "blackout", "disjoint", "overlap", "gaussian").
    block_size_mcar : int
        Size of blocks removed in MCAR

    Returns
    -------
    TimeSeries
        TimeSeries object containing contaminated data.
    """


    from imputegap.recovery.contamination import GenGap
    from imputegap.recovery.manager import TimeSeries

    pattern_low = pattern.lower()
    ptn = pattern_low.replace('_', '').replace('-', '')

    if ptn == "mcar" or ptn == "missing_completely_at_random":
        incomp_data = GenGap.mcar(input_data=ts.data, rate_dataset=dataset_rate, rate_series=series_rate, block_size=block_size, offset=offset, seed=seed, explainer=explainer, logic_by_series=logic_by_series, verbose=verbose)
    elif ptn == "mp" or ptn == "missingpercentage" or ptn == "aligned":
        incomp_data = GenGap.aligned(input_data=ts.data, rate_dataset=dataset_rate, rate_series=series_rate, offset=offset, explainer=explainer, logic_by_series=logic_by_series, verbose=verbose)
    elif ptn == "ps" or ptn == "percentageshift" or ptn == "scattered" or ptn == "scatter":
        incomp_data = GenGap.scattered(input_data=ts.data, rate_dataset=dataset_rate, rate_series=series_rate, offset=offset, seed=seed, explainer=explainer, logic_by_series=logic_by_series, verbose=verbose)
    elif ptn == "disjoint":
        incomp_data = GenGap.disjoint(input_data=ts.data, rate_series=dataset_rate, limit=1, offset=offset, logic_by_series=logic_by_series, verbose=verbose)
    elif ptn == "overlap":
        incomp_data = GenGap.overlap(input_data=ts.data, rate_series=dataset_rate, limit=limit, shift=shift, offset=offset, logic_by_series=logic_by_series, verbose=verbose)
    elif ptn == "gaussian":
        incomp_data = GenGap.gaussian(input_data=ts.data, rate_dataset=dataset_rate, rate_series=series_rate, std_dev=std_dev, offset=offset, seed=seed, explainer=explainer, logic_by_series=logic_by_series, verbose=verbose)
    elif ptn == "distribution" or pattern == "dist":
        incomp_data = GenGap.distribution(input_data=ts.data, rate_dataset=dataset_rate, rate_series=series_rate, probabilities_list=probabilities, offset=offset, seed=seed, explainer=explainer, logic_by_series=logic_by_series, verbose=verbose)
    elif ptn == "blackout":
        incomp_data = GenGap.blackout(input_data=ts.data, rate_series=dataset_rate, offset=offset, logic_by_series=logic_by_series, verbose=verbose)
    else:
        raise ValueError(f"\n(CONT) Pattern '{pattern}' not recognized, please choose your algorithm on this list :\n\t{TimeSeries().patterns}\n")
        incomp_data = None

    return incomp_data


def config_forecaster(model, params):
        """
        Configure and execute forecaster model for downstream analytics

        Parameters
        ----------
        model : str
            name of the forcaster model
        params : list of params
            List of paramaters for a forcaster model

        Returns
        -------
        Forecaster object (SKTIME/DART)
            Forecaster object for downstream analytics
        """

        from imputegap.recovery.manager import TimeSeries

        model_low = model.lower()
        mdl = model_low.replace('_', '').replace('-', '')

        if mdl == "prophet":
            from sktime.forecasting.fbprophet import Prophet
            forecaster = Prophet(**params)
        elif mdl == "expsmoothing":
            from sktime.forecasting.exp_smoothing import ExponentialSmoothing
            forecaster = ExponentialSmoothing(**params)
        elif mdl == "nbeats":
            from darts.models import NBEATSModel
            forecaster = NBEATSModel(**params)
        elif mdl == "xgboost":
            from darts.models.forecasting.xgboost import XGBModel
            forecaster = XGBModel(**params)
        elif mdl == "lightgbm":
            from darts.models.forecasting.lgbm import LightGBMModel
            forecaster = LightGBMModel(**params)
        elif mdl == "lstm":
            from darts.models.forecasting.rnn_model import RNNModel
            forecaster = RNNModel(**params)
        elif mdl == "deepar":
            from darts.models.forecasting.rnn_model import RNNModel
            forecaster = RNNModel(**params)
        elif mdl == "transformer":
            from darts.models.forecasting.transformer_model import TransformerModel
            forecaster = TransformerModel(**params)
        elif mdl == "hwadd":
            from sktime.forecasting.exp_smoothing import ExponentialSmoothing
            forecaster = ExponentialSmoothing(**params)
        elif mdl == "arima":
            from sktime.forecasting.arima import AutoARIMA
            forecaster = AutoARIMA(**params)
        elif mdl == "sf-arima" or mdl == "sfarima":
            from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
            forecaster = StatsForecastAutoARIMA(**params)
            forecaster.set_config(warnings='off')
        elif mdl == "bats":
            from sktime.forecasting.bats import BATS
            forecaster = BATS(**params)
        elif mdl == "ets":
            from sktime.forecasting.ets import AutoETS
            forecaster = AutoETS(**params)
        elif mdl == "croston":
            from sktime.forecasting.croston import Croston
            forecaster = Croston(**params)
        elif mdl == "theta":
            from sktime.forecasting.theta import ThetaForecaster
            forecaster = ThetaForecaster(**params)
        elif mdl == "unobs":
            from sktime.forecasting.structural import UnobservedComponents
            forecaster = UnobservedComponents(**params)
        elif mdl == "naive":
            from sktime.forecasting.naive import NaiveForecaster
            forecaster = NaiveForecaster(**params)
        else:
            raise ValueError(f"\n(DOWN) Forecasting model '{model}' not recognized, please choose your algorithm on this list :\n\t{TimeSeries().forecasting_models}\n")
            forecaster = None

        return forecaster



def get_resuts_unit_tests(algo_name, loader, verbose=True):
    """
    Returns (dataset, rmse, mae) for the given algo name
    from loader.toml.
    """
    try:
        import tomllib  # Python 3.11+
        with open(loader, "rb") as f:
            config = tomllib.load(f)
    except ImportError:
        import toml
        with open(loader, "r", encoding="utf-8") as f:
            config = toml.load(f)

    section = config[algo_name]

    dataset = section["dataset"]
    rmse = section["rmse"]
    mae = section["mae"]

    if verbose:
        print(f"\nloaded for {algo_name}: {dataset = }, {rmse = }, {mae = }\n")

    return dataset, rmse, mae


def window_truncation(feature_vectors, seq_len, stride=None, info="", verbose=True, deep_verbose=False):
    """
    Segment a sequence of feature vectors into fixed-length windows. In ImputeGAP, this is used in deep learning to reshape a 2D univariate dataset into a 3D windowed representation, enabling multivariate-like processing.
    See reconstruction_windowd_based() to restore the imputed matrix to its original shape.

    The code was inspired by: https://dl.acm.org/doi/10.1016/j.eswa.2023.119619

    Parameters
    ----------
    feature_vectors : np.ndarray
        Input array of feature vectors. Windowing is applied along the
        first axis (typically the time or sequence dimension).

    seq_len : int
        Length of each window (number of time steps per segment).

    stride : int, optional
        Step size between the starting indices of consecutive windows.
        Defaults to ``seq_len`` (non-overlapping windows).

    info : str, optional
        Additional descriptive string to include in the verbose log output.
        Defaults to an empty string.

    verbose : bool, optional
        If True, prints a summary of the computed windows (shape and
        configuration). Defaults to True.

    deep_verbose : bool, optional
        If True, prints the raw start indices used to generate the
        windows. Useful for debugging. Defaults to False.


    Returns
    -------
    np.ndarray
        Array of shape ``(num_windows, seq_len, features)`` containing the
        extracted windows, cast to ``float32``.
    """

    stride = seq_len if stride is None else stride
    values = feature_vectors.shape[0]
    start_indices = np.asarray(range(values // stride)) * stride

    if deep_verbose:
        print(f"{start_indices = }")

    sample_collector = []
    for idx in start_indices:
        if (idx + seq_len) > values:
            break
        sample_collector.append(feature_vectors[idx: idx + seq_len])

    dataset_strat_windows = np.asarray(sample_collector).astype('float32')
    if verbose:
        print(f"\t{info} windows have been computed ({seq_len=} | {stride=}): {dataset_strat_windows.shape}")

    return dataset_strat_windows


def dataset_add_dimensionality(matrix, seq_length=24, reshapable=True, adding_nans=True, three_dim=True, window=False, verbose=False, deep_verbose=False):
    """
    Prepare a 2D matrix for sequence-based models (sample strategy) by padding and optional reshaping to 3D.

    Parameters
    ----------
    matrix : np.ndarray
        Input 2D array of shape ``(N, M)``, where ``N`` is the number of
        time steps (rows) and ``M`` is the number of features (columns).

    seq_length : int, optional
        Target sequence length (number of time steps per segment). Used
        for padding and reshaping. Default is 24.

    reshapable : bool, optional
        If True, the matrix is padded (if needed) so that its number of
        rows is divisible by ``seq_length``. If False, sequences are
        extracted in non-overlapping chunks of length ``seq_length``
        without padding. Default is True.

    adding_nans : {True, False, None}, optional
        Controls the padding values:
        - None: pad with zeros.
        - True: pad with NaNs.
        - False: pad with per-column means (ignoring NaNs).
        Default is True (pad with NaNs).

    three_dim : bool, optional
        If True and ``reshapable`` is True, the padded matrix is reshaped
        to a 3D array of shape ``(num_sequences, seq_length, M)``.
        If False, the function returns the padded 2D matrix.
        Ignored when ``window=True`` or ``reshapable=False``.
        Default is True.

    window : bool, optional
        If True, the function only appends a block of ``seq_length`` rows
        (using the chosen padding strategy) and returns the resulting 2D
        matrix without reshaping. Default is False.

    verbose : bool, optional
        If True, prints information about padding and the resulting
        shape(s). Default is False.

    deep_verbose : bool, optional
        If True and ``three_dim`` is True, prints the full reshaped
        3D matrix for inspection. Default is False.

    Returns
    -------
    np.ndarray
          3D array of shape ``(N_padded // seq_length, seq_length, features)``.
    """
    if verbose:
        print(f"\ndataset is  pre-processed for 3 dimensionality, with a sequence length of {seq_length}.")

    N, M = matrix.shape

    if window:
        pad_len = seq_length
        if adding_nans is None:
            pad_block = np.full((pad_len, M), 0)
        else:
            pad_block = np.full((pad_len, M), np.nan)

        matrix = np.vstack([matrix, pad_block])
        if verbose:
            print(f"\tThe new shape is {matrix.shape}\n")
        return matrix

    if reshapable:
        # How many rows needed to make it divisible?
        remainder = N % seq_length
        if remainder != 0:
            pad_len = seq_length - remainder

            if adding_nans is None:
                if verbose:
                    print(f"the algorithm has added {pad_len} rows of NaNs")
                pad_block = np.full((pad_len, M), 0)  # fill with NaNs
            else:
                if adding_nans:
                    if verbose:
                        print(f"the algorithm has added {pad_len} rows of NaNs")
                    pad_block = np.full((pad_len, M), np.nan)  # fill with NaNs
                else:
                    col_mean = np.nanmean(matrix, axis=0)
                    if verbose:
                        print(f"the algorithm has added {pad_len} rows of {col_mean}")
                    pad_block = np.tile(col_mean, (pad_len, 1))  # repeat row of averages

            matrix = np.vstack([matrix, pad_block])
            N = matrix.shape[0]

        if not three_dim:
            if verbose:
                print(f"\tThe new shape is {matrix.shape}\n")
            return matrix

        # Now safe to reshape
        new_m = matrix.reshape(N // seq_length, seq_length, M)

        if verbose:
            print(f"\tThe new shape is {new_m.shape}\n")

        if deep_verbose:
            print(f"\tnew matrix : {new_m}\n")

        return new_m

    else:
        new_m = np.array([matrix[i:i + seq_length] for i in range(0, N - seq_length + 1, seq_length)])

        if verbose:
            print(f"\ntThe new shape is {new_m.shape}\n")

        return new_m
        

def dataset_reverse_dimensionality(matrix, expected_n: int, verbose: bool = True):
    """
    Convert (1, N, T, L) -> (N*T, L) or (N, T, L) -> (N*T, L), then trim to expected_n rows.

    Steps:
      1) If ndim==4, squeeze axis 0 (requires S==1).
      2) Reshape first two dims together -> (N*T, L).
      3) Drop the last (N*T - expected_n) rows.

    Args:
        matrix: np.ndarray of shape (1, N, T, L) or (N, T, L)
        expected_n: final number of rows after trimming (e.g., 1000)
        verbose: print shapes and removed-row count

    Returns:
        np.ndarray of shape (expected_n, L)
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"'matrix' must be a numpy array, got {type(matrix)}")

    if verbose:
        print("\nThe dataset will be reverse back to its original dimensionality...")
    # 1) Optional squeeze if 4D
    if matrix.ndim == 4:
        S, N, T, L = matrix.shape
        if verbose:
            print(f"\tinput: {matrix.shape} (S={S}, N={N}, T={T}, L={L})")
        if S != 1:
            raise ValueError(f"\tCannot squeeze: expected S==1 on the first dim, got S={S}")
        imp = np.squeeze(matrix, axis=0)     # (N, T, L)
        if verbose:
            print(f"\tafter squeeze -> {imp.shape}")
    elif matrix.ndim == 3:
        N, T, L = matrix.shape
        if verbose:
            print(f"\tinput: {matrix.shape} (N={N}, T={T}, L={L})")
        imp = matrix
    else:
        raise ValueError(f"\tExpected a 3D or 4D array, got {matrix.ndim}D with shape {matrix.shape}")

    # 2) Reshape (N, T, L) -> (N*T, L)
    imp = imp.reshape(N * T, L)
    if verbose:
        print(f"\tafter reshape -> {imp.shape} (N*T={N*T}, L={L})")

    # 3) Trim to expected_n rows
    total_rows = N * T
    if expected_n < 0:
        raise ValueError(f"\texpected_n must be non-negative, got {expected_n}")
    if expected_n > total_rows:
        raise ValueError(f"\texpected_n ({expected_n}) > total rows ({total_rows}) after reshape")

    removed = total_rows - expected_n
    if removed > 0:
        imp = imp[:-removed, :]
    if verbose:
        print(f"\tafter trim -> {imp.shape} (removed {removed} rows)")

    return imp



def __marshal_as_numpy_column(__ctype_container, __py_sizen, __py_sizem):
    """
    Marshal a ctypes container as a numpy column-major array.

    Parameters
    ----------
    __ctype_container : ctypes.Array
        The input ctypes container (flattened matrix).
    __py_sizen : int
        The number of rows in the numpy array.
    __py_sizem : int
        The number of columns in the numpy array.

    Returns
    -------
    numpy.ndarray
        A numpy array reshaped to the original matrix dimensions (row-major order).
    """
    __numpy_marshal = __numpy_import.array(__ctype_container).reshape(__py_sizem, __py_sizen).T;

    return __numpy_marshal;


def __marshal_as_native_column(__py_matrix):
    """
    Marshal a numpy array as a ctypes flat container for passing to native code.

    Parameters
    ----------
    __py_matrix : numpy.ndarray
        The input numpy matrix (2D array).

    Returns
    -------
    ctypes.Array
        A ctypes array containing the flattened matrix (in column-major order).
    """
    __py_input_flat = __numpy_import.ndarray.flatten(__py_matrix.T);
    __ctype_marshal = __numpy_import.ctypeslib.as_ctypes(__py_input_flat);

    return __ctype_marshal;


def display_title(title="Master Thesis", aut="Quentin Nater", lib="ImputeGAP", university="University Fribourg"):
    """
    Display the title and author information.

    Parameters
    ----------
    title : str, optional
        The title of the thesis (default is "Master Thesis").
    aut : str, optional
        The author's name (default is "Quentin Nater").
    lib : str, optional
        The library or project name (default is "ImputeGAP").
    university : str, optional
        The university or institution (default is "University Fribourg").

    Returns
    -------
    None
    """

    print("=" * 100)
    print(f"{title} : {aut}")
    print("=" * 100)
    print(f"    {lib} - {university}")
    print("=" * 100)


def auto_seq_llms(data_x, goal="seq", subset=False, high_limit=200, exception=False, b=None, verbose=True, deep_verbose=False):
    """
    Brute-force search for nice (seq_len, batch_size) pairs.

    If subset is False:
        data_x: array of shape (T, ...)

    If subset is True:
        We internally split T into train / test / val using 0.7 / 0.2 / 0.1
        and ensure that batch_size is <= num_windows for *each* subset.

    Returns:
        (seq_len, batch_size)
    """

    T = data_x.shape[0]
    F = data_x.shape[1]
    max_batch = T // 2

    if b is not None:
        max_batch = F//2

    if high_limit > T*0.1 and not exception:
        high_limit = int(T*0.1)

    if T > 50:
        start_batch = 8
    else:
        start_batch = 2
    if b is not None:
        start_batch = 2

    if subset:
        Tr = int(T * 0.7)
        sizes = [T, Tr]
    else:
        Tr = T//3
        sizes = [T, Tr]

    starting_point = max(2, (T // 2) - 1)
    starting_point = min(starting_point, high_limit)

    candidates = []

    for seq_len in range(starting_point, 1, -1):
        list_windows = []
        valid_seq = True

        if seq_len % 2 == 1:
            if seq_len != 1:
                continue

        for s in sizes:
            num_windows = s - seq_len + 1
            if num_windows < 2:
                valid_seq = False
                break
            list_windows.append(num_windows)

        if not valid_seq:
            continue

        for batch_size in range(start_batch, max_batch + 1):
            # *** FIX: iterate over list_windows, not max_possible_batch ***
            remainders=[]
            for nw in list_windows:
                r = nw % batch_size
                nbr = nw // batch_size
                if nbr > 1:
                    nbr = 0
                if nbr == 1:
                    nbr = 1
                r = r + nbr
                remainders.append(r)

            total_remainder = sum(remainders)

            if deep_verbose:
                print(f"{seq_len = } | {batch_size = }: {total_remainder = }")

            # ------- scoring logic -------
            if goal == "seq":
                # prefer perfect match first, then larger seq_len
                score = 1000 * (total_remainder > 0) - (2*seq_len) + abs(seq_len - batch_size)
            elif goal == "batch":
                # prefer perfect match first, then larger batch_size
                score = 1000 * (total_remainder > 0) - (2*batch_size) + abs(seq_len - batch_size)
            else:  # "balance"
                # keep original spirit, but on aggregated remainder
                score = total_remainder + abs(seq_len - batch_size) * 0.1

            # store a copy of list_windows for this candidate
            candidates.append((score, seq_len, batch_size, list_windows.copy()))

    if not candidates:
        if verbose:
            print("\ncompute pre-processor → No valid (seq_len, batch_size) combinations found.\n")
        return None, None

    # pick the combination with the lowest score
    candidates.sort(key=lambda x: x[0])
    best = candidates[0]
    score, seq_len, batch_size, best_windows = best

    if verbose:
        if subset:
            print(f"\ncompute pre-processor {score=} → Best seq_len={seq_len}, batch_size={batch_size}, num_windows_per_subset={best_windows}\n")
        else:
            num_windows = best_windows[0]
            print(f"\ncompute pre-processor {score=} → Best seq_len={seq_len}, batch_size={batch_size} (num_windows={num_windows})\n")

    return seq_len, batch_size


def auto_seq_sample(matrix, tr_ratio, high_val=98, verbose=True):
    """
    Automatically select a suitable sequence length and batch size
    based on the dataset size and a predefined batch-size table.

    The function iteratively searches for an even `seq_len`, starting from
    `high_val` and decreasing by 2, until it is less than or equal to
    `small_set`, where:

        small_set = int(T * (1 - tr_ratio)) // 2

    with `T` being the number of time steps (rows) in `matrix`.
    If the search goes below 2, `seq_len` is clamped to 2.

    Once `seq_len` is found, the batch size is chosen from a fixed
    table `[2, 4, 8, 16, 32, 64, 96]` as the value closest to `seq_len`.

    Parameters
    ----------
    matrix : np.ndarray
        Input 2D array of shape (T, F), where T is the number of time steps
        and F the number of features.

    tr_ratio : float
        Training ratio in [0, 1]. Used to compute the size of the
        "smallest set" (typically validation/test portion) that `seq_len`
        should not exceed.

    high_val : int, optional
        Initial (maximum) candidate sequence length from which the search
        starts and decreases by 2. Default is 98.

    verbose : bool, optional
        If True, prints the selected `seq_len`, `batch_size` and
        the computed `small_set`. Default is True.


    Returns
    -------
    seq_len : int
        Selected sequence length, guaranteed to be at least 2 and
        less than or equal to `small_set`.
    batch_size : int
        Selected batch size from the fixed table `[2, 4, 8, 16, 32, 64, 96]`
        that is closest (in absolute difference) to `seq_len`.
    """
    T, F = matrix.shape
    found = False

    batch_table = [2, 4, 8, 16, 32, 64, 96]

    small_set = int(T*(1-tr_ratio))//2

    if small_set <= 1000:
        high_val = 50
        batch_table = [2, 4, 8, 16, 32]
    if small_set <= 200:
        high_val = 26
        batch_table = [2, 4, 8, 16]

    seq_len = high_val

    while found is False:
        seq_len = seq_len - 2
        if seq_len <= small_set:
            found = True
        if seq_len <= 2:
            seq_len = 2
            break

    batch_size = min(batch_table, key=lambda b: abs(b - seq_len))

    if verbose:
        print(f"\nthe seq_len found is {seq_len}, and the batch_size {batch_size}, to match with the smallest set {small_set}\n")

    return seq_len, batch_size


def reconstruction_window_based(preds, nbr_timestamps, sliding_windows=1, verbose=True, deep_verbose=False):
    """
    Reconstruct the full time series after window-based imputation. This function restores the original univariate series or 2D matrix from the 3D windowed (multivariate-style) representation used during the deep learning process.
    See window_truncation() for the preprocessing transformation applied beforehand.

    Parameters
    ----------
    preds : torch.Tensor
        Predicted windows of shape ``(N, L, F)``, where:
        - ``N`` is the number of windows,
        - ``L`` is the window length (sequence length),
        - ``F`` is the number of features per time step.

    nbr_timestamps : int
        Target length ``T`` of the reconstructed time series along the
        time dimension (number of time steps).

    sliding_windows : int, optional
        Step size between the starting indices of consecutive windows in
        the original time series. The i-th window is placed starting at
        index ``i * sliding_windows``. Default is 1.

    verbose : bool, optional
        If True, prints a summary of the reconstruction process and basic
        completeness statistics. Default is True.

    deep_verbose : bool, optional
        If True, prints detailed information about the index ranges used
        for each window and the internal count matrix. Useful for
        debugging. Default is False.

    Returns
    -------
    torch.Tensor
        Reconstructed time series of shape ``(T, D)``, where
        overlapping windows have been averaged at each time step.
    """
    import torch
    N, L, F = preds.shape
    T = nbr_timestamps

    if verbose:
        print(f"\nreconstruction of the windows shaped matrix...\n\tsetup : {N =}, {L =}, {F =} -> {T = } : ", sep=" ", end=" ")
    recons = torch.zeros(T, F)
    counts = torch.zeros(T, F)
    for i in range(N):
        start = i * sliding_windows
        seq = (start + (preds[i].shape[0]))
        if deep_verbose:
            print(f"{i}-{seq - 1}|", sep="", end="")
        recons[start:seq] += preds[i]
        counts[start:seq] += 1

    if deep_verbose:
        print(f"{counts = }")

    mask_l = counts > 0
    recons[mask_l] = recons[mask_l] / counts[mask_l]
    # =test===================================================================================================
    row_sums = recons.sum(dim=1)  # if recons is also a torch.Tensor
    mask_nonzero = ~torch.isclose(row_sums, torch.tensor(0.0))
    full = mask_nonzero.all().item()

    rows_all_at_least_one = (counts >= 1).all(dim=1)
    num_bad_rows = (~rows_all_at_least_one).sum().item()
    bad_values = (~mask_nonzero).sum().item()

    if verbose:
        if full and num_bad_rows == 0:
            print(f"the reconstruction has been done successfully, full recovery matrix.\n"
                f"\tnumber of time steps reconstructed: {nbr_timestamps - num_bad_rows}/{nbr_timestamps}, "
                f"number of values not handled: {bad_values}")
        else:
            print(f"the reconstruction has been done successfully, full recovery matrix.\n"
                f"\tnumber of time steps reconstructed: {nbr_timestamps - num_bad_rows}/{nbr_timestamps}, "
                f"number of values not handled: {bad_values}")
    # ======================================================================================================
    return recons

def check_contamination_series(ts_m, algo="the algorithm", verbose=True):
    """
    Verify whether the input time series matrix meets the contamination constraints
    required by uni-dimensional algorithms (such as SPIRIT).

    Specifically, this function checks if only the first series (column 0) contains
    missing (NaN) values. If any other series is contaminated, it reports an
    imputation error (optionally printing a message) and returns `True` to signal
    that an issue exists.

    Parameters
    ----------
    ts_m : np.ndarray
        A 2D NumPy array representing the time series matrix, where each column
        corresponds to a separate series.
    algo : str, optional
        The name of the algorithm being validated. Used only for logging in
        the printed error message. Default is "the algorithm".
    verbose : bool, optional
        If True, prints an error message when contamination is detected outside
        of series 0. Default is True.

    Returns
    -------
    bool
        False if only series 0 is contaminated (valid input).
        True if contamination exists in any other series (invalid input).
    """
    nan_counts_per_col = np.sum(np.isnan(ts_m), axis=0)
    cols_with_nans = np.where(nan_counts_per_col > 0)[0].shape[0]

    if nan_counts_per_col[0] > 0 and np.sum(nan_counts_per_col > 0) == 1:
        return False
    else:
        if verbose:
            print(f"(IMPUTATION-ERROR) {algo} is a uni-dimensional algorithm and can only operate when series 0 is the sole contaminated one.\n\tThe provided matrix contains {cols_with_nans} contaminated series.\n")
        return True


def search_path(set_name="test"):
    """
    Find the accurate path for loading test files.

    Parameters
    ----------
    set_name : str, optional
        Name of the dataset (default is "test").

    Returns
    -------
    str
        The correct file path for the dataset.
    """

    if set_name in list_of_datasets():
        return set_name + ".txt"
    else:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(here, "datasets/" + set_name)
        if not os.path.exists(filepath):
            filepath = "../imputegap/datasets/" + set_name
            if not os.path.exists(filepath):
                filepath = filepath[1:]
        return filepath


def get_missing_ratio(incomp_data):
    """
    Check whether the proportion of missing values in the contaminated data is acceptable
    for training a deep learning model.

    Parameters
    ----------
    incomp_data : TimeSeries (numpy array)
            TimeSeries object containing dataset.

    Returns
    -------
    bool
        True if the missing data ratio is less than or equal to 40%, False otherwise.
    """
    import numpy as np

    miss_m = incomp_data
    total_values = miss_m.size
    missing_values = np.isnan(miss_m).sum()
    missing_ratio = missing_values / total_values

    return missing_ratio


def verification_limitation(percentage, low_limit=0.001, high_limit=1.0):
    """
    Format and verify that the percentage given by the user is within acceptable bounds.

    Parameters
    ----------
    percentage : float
        The percentage value to be checked and potentially adjusted.
    low_limit : float, optional
        The lower limit of the acceptable percentage range (default is 0.01).
    high_limit : float, optional
        The upper limit of the acceptable percentage range (default is 1.0).

    Returns
    -------
    float
        Adjusted percentage based on the limits.

    Raises
    ------
    ValueError
        If the percentage is outside the accepted limits.

    Notes
    -----
    - If the percentage is between 1 and 100, it will be divided by 100 to convert it to a decimal format.
    - If the percentage is outside the low and high limits, the function will print a warning and return the original value.
    """
    if low_limit <= percentage <= high_limit:
        return percentage  # No modification needed
    elif 1 <= percentage <= 100:
        print(f"The percentage {percentage} is between 1 and 100. Dividing by 100 to convert to a decimal.")
        return percentage / 100
    else:
        raise ValueError(f"The percentage {percentage} is out of the acceptable range.")


def dl_integration_transformation(input_matrix, tr_ratio=0.8, inside_tr_cont_ratio=0.2, split_ts=1, split_val=0, nan_val=-99999, prevent_leak=True, offset=0.05, block_selection=True, seed=42, verbose=False):
    """
        Prepares contaminated data and corresponding masks for deep learning-based imputation training,
        validation, and testing.

        This function simulates missingness in a controlled way, optionally prevents information leakage,
        and produces masks for training, testing, and validation using different contamination strategies.

        Parameters:
        ----------
        input_matrix : np.ndarray
            The complete input time series data matrix of shape [T, N] (time steps × variables).

        tr_ratio : float, default=0.8
            The fraction of data to reserve for training when constructing the test contamination mask.

        inside_tr_cont_ratio : float, default=0.2
            The proportion of values to randomly drop inside the training data for internal contamination.

        split_ts : float, default=1
            Proportion of the total contaminated data assigned to the test set.

        split_val : float, default=0
            Proportion of the total contaminated data assigned to the validation set.

        nan_val : float, default=-99999
            Value used to represent missing entries in the masked matrix.
            nan_val=-1 can be used to set mean values

        prevent_leak : bool, default=True
            Replace the value of NaN with a high number to prevent leakage.

        offset : float, default=0.05
            Minimum temporal offset in the begining of the series

        block_selection : bool, default=True
            Whether to simulate missing values in contiguous blocks (True) or randomly (False).

        seed : int, default=42
            Seed for NumPy random number generation to ensure reproducibility.

        verbose : bool, default=False
            Whether to print logging/debug information during execution.

        Returns:
        -------
        cont_data_matrix : np.ndarray
            The input matrix with synthetic missing values introduced.

        mask_train : np.ndarray
            Boolean mask of shape [T, N] indicating the training contamination locations (True = observed, False = missing).

        mask_test : np.ndarray
            Boolean mask of shape [T, N] indicating the test contamination locations.

        mask_valid : np.ndarray
            Boolean mask of shape [T, N] indicating the validation contamination locations.

        error : bool
            Tag which is triggered if the operation is impossible.
    """

    cont_data_matrix = input_matrix.copy()
    original_missing_ratio = get_missing_ratio(cont_data_matrix)

    cont_data_matrix, new_mask, error = prepare_testing_set(incomp_m=cont_data_matrix, original_missing_ratio=original_missing_ratio, block_selection=block_selection, tr_ratio=tr_ratio, verbose=verbose)

    if prevent_leak:
        if nan_val == -1:
            import numpy as np
            nan_val = np.nanmean(input_matrix)
            print(f"\nNaN replacement Mean Value : {nan_val}\n")
        cont_data_matrix = prevent_leakage(cont_data_matrix, new_mask, nan_val, verbose)

    mask_test, mask_valid, nbr_nans = split_mask_bwt_test_valid(cont_data_matrix, test_rate=split_ts, valid_rate=split_val, nan_val=nan_val, verbose=verbose, seed=seed)
    mask_train = generate_random_mask(gt=cont_data_matrix, mask_test=mask_test, mask_valid=mask_valid, droprate=inside_tr_cont_ratio, offset=offset, verbose=verbose, seed=seed)

    return cont_data_matrix, mask_train, mask_test, mask_valid, error


def prepare_fixed_testing_set(incomp_m, tr_ratio=0.8, offset=0.05, block_selection=True, verbose=True):
    """
    Introduces additional missing values (NaNs) into a data matrix to match a specified training ratio.

    This function modifies a copy of the input matrix `incomp_m` by introducing NaNs
    such that the proportion of observed (non-NaN) values matches the desired `tr_ratio`.
    It returns the modified matrix and the corresponding missing data mask.

    Parameters
    ----------
    incomp_m : np.ndarray
       A 2D NumPy array with potential pre-existing NaNs representing missing values.

    tr_ratio : float
       Desired ratio of observed (non-NaN) values in the output matrix. Must be in the range (0, 1).

    offset : float
        Protected zone in the begining of the series

    block_selection : bool
        Select the missing values by blocks or randomly (True, is by block)

    verbose : bool
        Whether to print debug info.

    Returns
    -------
    data_matrix_cont : np.ndarray
       The modified matrix with additional NaNs introduced to match the specified training ratio.

    new_mask : np.ndarray
       A boolean mask of the same shape as `data_matrix_cont` where True indicates missing (NaN) entries.

    Raises
    ------
    AssertionError:
       If the final observed and missing ratios deviate from the target by more than 1%.

    Notes
    -----
        - The function assumes that the input contains some non-NaN entries.
        - NaNs are added in row-major order from the list of available (non-NaN) positions.
    """

    import numpy as np

    data_matrix_cont = incomp_m.copy()

    target_ratio = 1 - tr_ratio
    total_values = data_matrix_cont.size
    target_n_nan = int(target_ratio * total_values)

    # 2) Current number of NaNs
    current_n_nan = np.isnan(data_matrix_cont).sum()
    n_new_nans = target_n_nan - current_n_nan

    available_mask = ~np.isnan(data_matrix_cont)

    offset_vals = int(offset * data_matrix_cont.shape[1])
    for row in range(data_matrix_cont.shape[0]):
        available_mask[row, :offset_vals] = False  # protect leftmost `offset` columns in each row

    available_indices = np.argwhere(available_mask)

    # 3) Pick indices to contaminate
    if n_new_nans > 0:
        if block_selection :
            chosen_indices = available_indices[:n_new_nans]
        else:
            np.random.seed(42)
            chosen_indices = available_indices[np.random.choice(len(available_indices), n_new_nans, replace=False)]

        for i, j in chosen_indices:
            data_matrix_cont[i, j] = np.nan

    # 4) check ratio
    n_total = data_matrix_cont.size
    n_nan = np.isnan(data_matrix_cont).sum()
    n_not_nan = n_total - n_nan

    # Compute actual ratios
    missing_ratio = n_nan / n_total
    observed_ratio = n_not_nan / n_total

    # Check if they match expectations (within a small tolerance)
    assert abs(missing_ratio - target_ratio) < 0.01, f"Missing ratio {missing_ratio} is not {target_ratio}"
    assert abs(observed_ratio - tr_ratio) < 0.01, f"Missing ratio {observed_ratio} is not {tr_ratio}"

    # Create the new mask
    new_mask = np.isnan(data_matrix_cont)
    new_m = data_matrix_cont.copy()

    if verbose:
        print(f"(DL): TEST-SET > Test set fixed to {int(round(target_ratio*100))}% of the dataset, for {target_n_nan} values, add test values: {n_new_nans}")

    return new_m, new_mask

def split_mask_bwt_test_valid(data_matrix, test_rate=0.8, valid_rate=0.2, nan_val=None, verbose=False, seed=42):
    """
    Dispatch NaN positions in data_matrix to test and validation masks only.

    Parameters
    ----------
    data_matrix : numpy.ndarray
        Input matrix containing NaNs to be split.

    test_rate : float
        Proportion of NaNs to assign to the test set (default is 0.8).

    valid_rate : float
        Proportion of NaNs to assign to the validation set (default is 0.2).
        test_rate + valid_rate must equal 1.0.

    verbose : bool
        Whether to print debug info.

    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple
        test_mask : numpy.ndarray
            Binary mask indicating positions of NaNs in the test set.

        valid_mask : numpy.ndarray
            Binary mask indicating positions of NaNs in the validation set.

        n_nan : int
            Total number of NaN values found in the input matrix.
    """
    import numpy as np

    assert np.isclose(test_rate + valid_rate, 1.0), "test_rate and valid_rate must sum to 1.0"

    if seed is not None:
        np.random.seed(seed)

    if nan_val is None:
        nan_mask = np.isnan(data_matrix)
    else:
        nan_mask = data_matrix == nan_val

    nan_indices = np.argwhere(nan_mask)
    np.random.shuffle(nan_indices)

    n_nan = len(nan_indices)
    n_test = int(n_nan * test_rate)
    n_valid = n_nan - n_test

    if verbose:
        print(f"\n(DL): MASKS > creating mask (testing, validation): Total NaNs = {n_nan}")
        print(f"(DL): TEST-MASK > creating mask: Assigned to test = {n_test}")
        print(f"(DL): VALID-MASK > creating mask: Assigned to valid = {n_valid}")

    test_idx = nan_indices[:n_test]
    valid_idx = nan_indices[n_test:]

    mask_test = np.zeros_like(data_matrix, dtype=np.uint8)
    mask_valid = np.zeros_like(data_matrix, dtype=np.uint8)

    mask_test[tuple(test_idx.T)] = 1
    mask_valid[tuple(valid_idx.T)] = 1

    if verbose:
        print(f"(DL): TEST-MASK > Test mask NaNs: {mask_test.sum()}")
        print(f"(DL): VALID-MASK > Valid mask NaNs: {mask_valid.sum()}\n")

    return mask_test, mask_valid, n_nan


def generate_random_mask(gt, mask_test, mask_valid, droprate=0.2, offset=None, series_like=True, verbose=False, seed=42):
    """
    Generate a random training mask over the non-NaN entries of gt, excluding positions
    already present in the test and validation masks.

    Parameters
    ----------
    gt : numpy.ndarray
        Ground truth data (no NaNs).
    mask_test : numpy.ndarray
        Binary mask indicating test positions.
    mask_valid : numpy.ndarray
        Binary mask indicating validation positions.
    droprate : float
        Proportion of eligible entries to include in the training mask.
    series_like : bool
        The mask must be set on free series
    offset : float
        Protect of not the offset of the dataset
    verbose : bool
        Whether to print debug info.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Binary mask indicating training positions.
    """
    import numpy as np

    assert gt.shape == mask_test.shape == mask_valid.shape, "All input matrices must have the same shape"

    if seed is not None:
        np.random.seed(seed)

    mask_test_tmp =  mask_test.astype(int)
    mask_valid_tmp =  mask_valid.astype(int)

    # Valid positions: non-NaN and not in test/valid masks
    num_offset = 0
    mask_offset = np.zeros_like(gt, dtype=np.uint8)

    # just the cell must be free to be picked
    if offset is not None:
        if offset > droprate:
            offset = droprate
        mask_offset[:, :int(offset * gt.shape[1])] = 1
        num_offset = np.sum(mask_offset)

    if series_like:
        row_test = mask_test_tmp.any(axis=1)
        row_valid = mask_valid_tmp.any(axis=1)
        mask_test_tmp[row_test, :] = 1
        mask_valid_tmp[row_valid, :] = 1

    occupied_mask = (mask_test_tmp + mask_valid_tmp + mask_offset).astype(bool)
    eligible_mask = (~np.isnan(gt)) & (~occupied_mask)
    eligible_indices = np.argwhere(eligible_mask)

    n_train = int(len(eligible_indices) * droprate) + int(num_offset*droprate)

    np.random.shuffle(eligible_indices)
    selected_indices = eligible_indices[:n_train]

    mask_train = np.zeros_like(gt, dtype=np.uint8)
    mask_train[tuple(selected_indices.T)] = 1

    if verbose:
        print(f"(DL): TRAIN-MASK > eligible entries: {len(eligible_indices)}")
        print(f"(DL): TRAIN-MASK > selected training entries: {n_train}\n")

    # Sanity check: no overlap between training and test masks
    overlap = np.logical_and(mask_train, mask_test).sum()
    assert overlap == 0, f"Overlap detected between training and test masks: {overlap} entries."

    # Sanity check: no overlap between training and test masks
    overlap = np.logical_and(mask_train, mask_valid).sum()
    assert overlap == 0, f"Overlap detected between training and test masks: {overlap} entries."

    if verbose:
        print(f"(DL): TRAIN-MASK > Train mask NaNs: {mask_train.sum()}\n")

    return mask_train

def prevent_leakage(matrix, mask, replacement=0, verbose=True):
    """
        Replaces missing values in a matrix to prevent data leakage during evaluation.

        This function replaces all entries in `matrix` that are marked as missing in `mask`
        with a specified `replacement` value (default is 0). It then checks to ensure that
        there are no remaining NaNs in the matrix and that at least one replacement occurred.

        Parameters
        ----------
        matrix : np.ndarray
            A NumPy array potentially containing missing values (NaNs).

        mask : np.ndarray
            A boolean mask of the same shape as `matrix`, where True indicates positions
            to be replaced (typically where original values were NaN).

        replacement : float or int, optional
            The value to use in place of missing entries. Defaults to 0.

        verbose : bool
            Whether to print debug info.

        Returns
        -------
        matrix : np.ndarray
            The matrix with missing entries replaced by the specified value.

        Raises
        ------
        AssertionError:
            If any NaNs remain in the matrix after replacement, or if no replacements were made.

        Notes
        -----
            - This function is typically used before evaluation to ensure the model does not
              access ground truth values where data was originally missing.
    """

    import numpy as np

    matrix[mask] = replacement

    assert not np.isnan(matrix).any(), "matrix still contains NaNs"
    assert (matrix == replacement).any(), "matrix does not contain any zeros"

    if verbose:
        print(f"\n(DL) Reset all testing matrix values to {replacement} to prevent data leakage.")

    return matrix

def prepare_testing_set(incomp_m, original_missing_ratio, block_selection=True, tr_ratio=0.8, verbose=True):
    import numpy as np

    error = False
    mask_original_nan = np.isnan(incomp_m)

    if verbose:
        print(f"\n(DL) TEST-SET : testing ratio to reach = {1-tr_ratio:.2%}")
        print(f"\n(DL) TEST-SET : original missing ratio = {original_missing_ratio:.2%}")
        print(f"(DL) TEST-SET : original missing numbers = {np.sum(mask_original_nan)}")

    if original_missing_ratio > 1-tr_ratio:
        print(f"\n(ERROR) The proportion of original missing values is too high and will corrupt the training set.\n\tPlease consider reducing the percentage contamination pattern [{original_missing_ratio:.2%}] or decreasing the training ratio [{tr_ratio:.2%}].\n")
        return incomp_m, mask_original_nan, True

    if abs((1-tr_ratio) - original_missing_ratio) > 0.01:
        new_m, new_mask = prepare_fixed_testing_set(incomp_m, tr_ratio, block_selection=block_selection, verbose=verbose)

        if verbose:
            print(f"(DL) TEST-SET : building of the test set to reach a fix ratio of {1 - tr_ratio:.2%}...")
            final_ratio = get_missing_ratio(new_m)
            print(f"(DL) TEST-SET : final artificially missing ratio for test set = {final_ratio:.2%}")
            print(f"(DL) TEST-SET : final number of rows with NaN values = {np.sum(np.isnan(new_m).any(axis=1))}")
            print(f"(DL) TEST-SET : final artificially missing numbers = {np.sum(new_mask)}\n")

    else:
        new_m = incomp_m
        new_mask = mask_original_nan.copy()

    return new_m, new_mask, error


"""
def set_dic_position_dl(mask_test, split_idx, verbose=False):
    position_dic_imputegap = {}  # {original_row_index: "train"|"val"|"test"}
    non_test_seen = 0  # position among NON-TEST rows
    inc_tr = 0
    inc_val = 0
    inc_test = 0
    for i, is_test in enumerate(mask_test):
        if is_test:
            position_dic_imputegap[i] = ("test", inc_test)
            inc_test += 1
        else:
            if non_test_seen < split_idx:
                position_dic_imputegap[i] = ("train", inc_tr)
                inc_tr += 1
            else:
                position_dic_imputegap[i] = ("val", inc_val)
                inc_val += 1
            non_test_seen += 1

    if verbose:
        print("\nIndices ImputeGAP Deep Learning Training with Patterns:")
        for k, v in position_dic_imputegap.items():
            set, inc = v
            print(f"\t{k}\t{set}\t{inc}")

    return position_dic_imputegap

def compute_seq_length(M):

    seq_length = 1
    if M > 5000:
        seq_length = 3000
    elif M > 3000:
        seq_length = 1400
    elif M > 2000:
        seq_length = 1000
    elif M > 1000:
        seq_length = 600
    elif M > 300:
        seq_length = 100
    elif M > 30:
        seq_length = 16
    else:
        if M % 5 == 0:
            seq_length = M // 5
        elif M % 6 == 0:
            seq_length = M // 6
        elif M % 2 == 0:
            seq_length = M // 2 - 2
            if seq_length < 1:
                seq_length = 1
        elif M % 3 == 0:
            seq_length = M // 3

    return seq_length


def compute_batch_size(data, min_size=4, max_size=16, divisor=2, verbose=True):
    
    M, N = data.shape

    batch_size = min(M // divisor, max_size)

    if batch_size < min_size:
        batch_size = min_size

    if batch_size % 2 != 0:
        batch_size = batch_size + 1
        if batch_size > max_size:
            batch_size = batch_size -2

    if batch_size < 1:
        batch_size = 1

    if verbose:
        print(f"(Batch-Size) Computed batch size: {batch_size}\n")

    return batch_size
"""


def load_share_lib(name="lib_cdrec", verbose=True):
    """
    Load the shared library based on the operating system.

    Parameters
    ----------
    name : str, optional
        The name of the shared library (default is "lib_cdrec").
    lib : bool, optional
        If True, the function loads the library from the default 'imputegap' path; if False, it loads from a local path (default is True).
    verbose : bool, optional
        Whether to display the contamination information (default is True).

    Returns
    -------
    ctypes.CDLL
        The loaded shared library object.
    """
    system = platform.system()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # mac os ===========================================================================================================
    if system == "Darwin":
        lib_path = importlib.resources.files('imputegap.algorithms.lib').joinpath("./" + str(name) + ".dylib")

        try:  # try inner file C++
            cpp_wrapper = ctypes.CDLL(lib_path)
            if verbose:
                print(f"\n(SYS) Wrapper files loaded for C++ : ", {lib_path}, "\n")
        except Exception:
            try:
                lib_path = importlib.resources.files('imputegap.algorithms.lib').joinpath("./" + str(name) + "_new.dylib")
                cpp_wrapper = ctypes.CDLL(lib_path)
                print(f"(SYS-UPD) C++ shared object linked with new version of armadillo: {lib_path}\n")
            except Exception:
                lib_path = os.path.join(here, 'algorithms/lib/' + name + ".dylib")
                cpp_wrapper = ctypes.CDLL(lib_path)
                print(f"(SYS-UPD) C++ shared object linked with the user  path : {lib_path}\n")
    # other ===========================================================================================================
    else:
        lib_path = importlib.resources.files('imputegap.algorithms.lib').joinpath("./" + str(name) + ".so")

        try:  # try inner file C++
            cpp_wrapper = ctypes.CDLL(lib_path)
            if verbose:
                print(f"\n(SYS) Wrapper files loaded for C++ : ", {lib_path}, "\n")
        except Exception:
            lib_path = os.path.join(here, 'algorithms/lib/' + name + ".so")
            cpp_wrapper = ctypes.CDLL(lib_path)
            print(f"(SYS-UPD) C++ shared object linked with the user path : {lib_path}\n")

    return cpp_wrapper



def control_boundaries(rank, boundary, algorithm="Algorithm", reduction=1):
    """
    Ensure that the rank does not exceed the boundary limit.

    Parameters
    ----------
    rank : int
        The input rank, typically representing the number of components or factors.
    boundary : int
        The maximum allowed value, usually corresponding to the number of available series.
    algorithm : str, optional
        The name of the algorithm using this control check (default is "Algorithm").
    reduction : int, optional
        The amount to reduce the boundary by if the rank exceeds it (default is 1).

    Returns
    -------
    int
        The adjusted rank value. If the input rank is valid, it is returned unchanged.
        If it exceeds the boundary, a reduced value is returned. If no valid reduction is
        possible, returns 1.
    """

    if rank >= boundary:
        new_estimator = boundary-reduction
        print(f"(ERROR) {algorithm}\n\trank {rank} is higher than the number of series {boundary}. Reduce to {new_estimator}.\n")

        if new_estimator > 0:
            return new_estimator
        else:
            print(f"(ERROR) {algorithm}\n\tNot enough series to impute with this algorithm {boundary} <= 0.\n")
            return 1
    else:
        return rank



def list_of_algorithms():
    """
    Return the list of available imputation algorithms.

    Parameters
    ----------
    None

    Returns
    -------
    list of str
       A sorted list of algorithm names supported by the framework.
    """
    return sorted([
        "CDRec",
        "IterativeSVD",
        "GROUSE",
        "ROSL",
        "SPIRIT",
        "SoftImpute",
        "SVT",
        "TRMF",
        "STMVL",
        "DynaMMo",
        "TKCM",
        "IIM",
        "XGBOOST",
        "MICE",
        "MissForest",
        "KNNImpute",
        "Interpolation",
        "MinImpute",
        "MeanImpute",
        "ZeroImpute",
        "MeanImputeBySeries",
        "MRNN",
        "BRITS",
        "DeepMVI",
        "MPIN",
        "PRISTI",
        "MissNet",
        "GAIN",
        "GRIN",
        "BayOTIDE",
        "HKMFT",
        "BitGraph",
        "SAITS",
        "NuwaTS",
        "GPT4TS",
        "TimesNet",
        "CSDI"
    ])

def list_of_patterns():
    """
    Return the list of available imputation patterns.

    Parameters
    ----------
    None

    Returns
    -------
    list of str
       A sorted list of patterns names supported by the framework.
    """
    return sorted([
        "aligned",
        "disjoint",
        "overlap",
        "scattered",
        "mcar",
        "gaussian",
        "distribution"
    ])

def list_of_datasets(txt=False):
    """
    Return the list of available datasets from ImputeGAP.

    Parameters
    ----------
    None

    Returns
    -------
    list of str
       A sorted list of datasets names supported by ImputeGAP.
    """
    list = sorted([
        "airq",
        "bafu",
        "chlorine",
        "climate",
        "drift",
        "eeg-alcohol",
        "eeg-reading",
        "electricity",
        #"fmri-stoptask",
        "forecast-economy",
        "meteo",
        "motion",
        "soccer",
        #"solar",
        "sport-activity",
        "stock-exchange",
        "temperature",
        "traffic"
    ])
    if txt:
        list = [dataset + ".txt" for dataset in list]
    return list



def list_of_optimizers():
    """
    Return the list of available optimizers from ImputeGAP.

    Parameters
    ----------
    None

    Returns
    -------
    list of str
       A sorted list of optimizers names supported by ImputeGAP.
    """
    return sorted([
        "ray_tune",
        "bayesian",
        "particle_swarm",
        "successive_halving",
        "greedy"
    ])

def list_of_downstreams():
    """
    Return the list of available downstream models from ImputeGAP.

    Parameters
    ----------
    None

    Returns
    -------
    list of str
       A sorted list of downstream models names supported by ImputeGAP.
    """
    return sorted(list_of_downstreams_sktime() + list_of_downstreams_darts())


def list_of_downstreams_sktime():
    """
    Return the list of available downstream models (sktime) from ImputeGAP.

    Parameters
    ----------
    None

    Returns
    -------
    list of str
       A sorted list of downstream models names supported by ImputeGAP.
    """
    return sorted([
        "prophet",
        "exp-smoothing",
        "hw-add",
        "arima",
        "sf-arima",
        #"bats",
        "ets",
        "croston",
        "theta",
        "unobs",
        "naive"
    ])

def list_of_downstreams_darts():
    """
    Return the list of available downstream models (darts) from ImputeGAP.

    Parameters
    ----------
    None

    Returns
    -------
    list of str
       A sorted list of downstream models names supported by ImputeGAP.
    """
    return sorted([
        "nbeats",
        "xgboost",
        "lightgbm",
        "lstm",
        "deepar",
        "transformer"
    ])

def list_of_extractors():
    """
    Return the list of available extractors from ImputeGAP.

    Parameters
    ----------
    None

    Returns
    -------
    list of str
       A sorted list of extractors names supported by ImputeGAP.
    """
    return sorted([
        "pycatch",
        "tsfel",
        "tsfresh"
    ])

def list_of_families():
    """
    Return the list of available families of imputation techniques from ImputeGAP.

    Parameters
    ----------
    None

    Returns
    -------
    list of str
       A sorted list of families of imputation techniques names supported by ImputeGAP.
    """
    return sorted(["DeepLearning", "MatrixCompletion", "PatternSearch", "MachineLearning", "Statistics", "LLMs"])

def list_of_metrics():
    """
    Return the list of available metrics from ImputeGAP.

    Parameters
    ----------
    None

    Returns
    -------
    list of str
       A sorted list of families of imputation metrics supported by ImputeGAP.
    """
    return ["RMSE", "MAE", "MI", "CORRELATION", "RUNTIME", "RUNTIME_LOG"]

def list_of_algorithms_deep_learning():
    """
    Returns all imputation algorithms of the Deep Learning family.
    """
    return list_of_algorithms_with_families(specify_family="DeepLearning")

def list_of_algorithms_matrix_completion():
    """
    Returns all imputation algorithms of the Matrix Completion family.
    """
    return list_of_algorithms_with_families(specify_family="MatrixCompletion")

def list_of_algorithms_pattern_search():
    """
    Returns all imputation algorithms of the Pattern Search family.
    """
    return list_of_algorithms_with_families(specify_family="PatternSearch")

def list_of_algorithms_machine_learning():
    """
    Returns all imputation algorithms of the Machine Learning family.
    """
    return list_of_algorithms_with_families(specify_family="MachineLearning")

def list_of_algorithms_statistics():
    """
    Returns all imputation algorithms of the Statistics family.
    """
    return list_of_algorithms_with_families(specify_family="Statistics")

def list_of_algorithms_llms():
    """
    Returns all imputation algorithms of the LLMs family.
    """
    return list_of_algorithms_with_families(specify_family="LLMs")

def list_of_algorithms_with_families(specify_family=None):
    """
    Return the list of available imputation techniques (with families) from ImputeGAP.

    Parameters
    ----------
    None

    Returns
    -------
    list of str
       A sorted list of imputation techniques (with families) supported by ImputeGAP.
    """

    my_list = [
        "MatrixCompletion.CDRec",
        "MatrixCompletion.IterativeSVD",
        "MatrixCompletion.GROUSE",
        "MatrixCompletion.ROSL",
        "MatrixCompletion.SPIRIT",
        "MatrixCompletion.SoftImpute",
        "MatrixCompletion.SVT",
        "MatrixCompletion.TRMF",
        "PatternSearch.STMVL",
        "PatternSearch.DynaMMo",
        "PatternSearch.TKCM",
        "MachineLearning.IIM",
        "MachineLearning.XGBOOST",
        "MachineLearning.MICE",
        "MachineLearning.MissForest",
        "Statistics.KNNImpute",
        "Statistics.Interpolation",
        "Statistics.MinImpute",
        "Statistics.MeanImpute",
        "Statistics.ZeroImpute",
        "Statistics.MeanImputeBySeries",
        "DeepLearning.MRNN",
        "DeepLearning.BRITS",
        "DeepLearning.DeepMVI",
        "DeepLearning.MPIN",
        "DeepLearning.PRISTI",
        "DeepLearning.MissNet",
        "DeepLearning.GAIN",
        "DeepLearning.GRIN",
        "DeepLearning.BayOTIDE",
        "DeepLearning.HKMFT",
        "DeepLearning.BitGraph",
        "DeepLearning.SAITS",
        "DeepLearning.CSDI",
        "DeepLearning.TimesNet",
        "LLMs.NuwaTS",
        "LLMs.GPT4TS"
    ]

    def normalize_family(family: str) -> str:
        return family.replace(" ", "").lower()

    if specify_family is not None:
        target = normalize_family(specify_family)
        return [algo.split(".")[1] for algo in my_list if normalize_family(algo.split(".")[0]) == target]
    else:
        return sorted(my_list)

def list_of_normalizers():
    """
    Return the list of available normalizer (with families) from ImputeGAP.

    Parameters
    ----------
    None

    Returns
    -------
    list of str
       A sorted list of normalizer supported by ImputeGAP.
    """

    return ["z_score", "min_max"]


def clean_missing_values(raw_data=None, substitute="zero", mask=None):
    """
    Replace all NaN values in a 2D matrix by a column-wise substitute.

    Parameters
    ----------
    raw_data : np.ndarray
        2D input array of shape (N, M) containing missing values encoded

    substitute : {"mean", "median", "zero"}, optional
        Strategy used to replace NaNs per column:
        - "mean":   replace NaNs with the column-wise mean (ignoring NaNs).
        - "median": replace NaNs with the column-wise median (ignoring NaNs).
        - "zero":   replace NaNs with 0.
        Default is "mean".

    mask, np.ndarraym optional
        Replace the normal NaNs detection

    Returns
    -------
    np.ndarray
        2D array of shape (N, M) with NaNs replaced column-wise
    """
    if raw_data is None:
        raise ValueError("raw_data must not be None.")

    if mask is None:
        mask = np.isnan(raw_data)

    if not mask.any():
        return raw_data.copy()

    n_rows, n_cols = raw_data.shape

    if substitute == "mean":
        col_values = np.nanmean(raw_data, axis=0)
    elif substitute == "median":
        col_values = np.nanmedian(raw_data, axis=0)
    elif substitute == "zero":
        col_values = np.zeros(n_cols, dtype=float)
    else:
        raise ValueError(f"Unknown substitute strategy '{substitute}'. Use 'mean', 'median', or 'zero'.")

    filled = raw_data.copy()
    rows, cols = np.where(mask)
    filled[rows, cols] = col_values[cols]

    return filled

def handle_nan_input(raw_data, incomp_data):
    raw_mask = 1-np.isnan(raw_data)
    ts_mask = 1-np.isnan(incomp_data)

    diff_raw = 1 - raw_mask
    imputed_mask = diff_raw + ts_mask

    return 1-imputed_mask


def prepare_deep_learning_params(incomp_data, seq_len, batch_size, sliding_windows, tr_ratio, verbose):

    error = False
    if sliding_windows == 0:
        multivariate = True
    else:
        multivariate = False

    if seq_len == -1 or batch_size == -1:
        if multivariate:
            seq_len, batch_size = auto_seq_sample(matrix=incomp_data, tr_ratio=tr_ratio, verbose=verbose)
            sliding_windows = seq_len
        else:
            seq_len, batch_size = auto_seq_llms(data_x=incomp_data, goal="seq", subset=True, high_limit=50, verbose=verbose)

    if seq_len > len(incomp_data):
        print(f"(ERROR) The current seq_length {seq_len} is not adapted to the contaminated matrix {len(incomp_data)} !")
        error =True

    return multivariate, seq_len, batch_size, sliding_windows, error