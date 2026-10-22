import ctypes
import os

import numpy as np
import pandas as pd
import toml
import importlib.resources
import numpy as __numpy_import
import platform




def load_parameters(query: str = "default", algorithm: str = "cdrec", dataset: str = "chlorine", optimizer: str = "b", path=None, verbose=False, task="default"):
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
        Optimizer type for optimal parameters (default is "b").

    path : str, optional
        Custom file path for the TOML file (default is None).

    verbose : bool, optional
        Whether to display the contamination information (default is False).

    task: string, optional
        Task of the model (default is "default"). It will load the default parameters for the specific task.
        "default" for regular imputation / "classification" for classification. / "forecasting" for forecasting.

    Returns
    -------
    dict, tuple, or None
        Algorithm-specific parameters loaded from the configuration file.
        The return type depends on the selected algorithm.
    """

    file_to_load = "default_values.toml"

    if task != "default" or task == "imputation":
        if task == "classification":
            file_to_load = "default_values_c.toml"
        else:
            file_to_load = "default_values_f.toml"

    if query == "default":
        if path is None:
            filepath = importlib.resources.files('imputegap.env').joinpath("./"+file_to_load)
        else:
            filepath = path
        if not os.path.exists(filepath):
            print(f"\n\tthe selected path is wrong, auto-redirection...")
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            filepath = os.path.join(here, "env/"+file_to_load)

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

    # IMPUTATION ================================================================

    if algorithm == "cdrec":
        rank = int(config[algorithm]['rank'])
        epsilon = float(config[algorithm]['epsilon'])
        iteration = int(config[algorithm]['iteration'])
        return {"rank":rank, "epsilon":epsilon, "iteration":iteration}


    elif algorithm == "seminar":
        imputer = str(config[algorithm]['imputer'])
        return {"imputer":imputer}

    elif algorithm == "new_imputer":
        imputer = str(config[algorithm]['imputer'])
        return {"imputer":imputer}

    elif algorithm == "stmvl":
        window_size = int(config[algorithm]['window_size'])
        gamma = float(config[algorithm]['gamma'])
        alpha = int(config[algorithm]['alpha'])
        return {"window_size":window_size, "gamma":gamma, "alpha":alpha}


    elif algorithm == "iim":
        learning_neighbors = int(config[algorithm]['learning_neighbors'])
        if query == "default":
            algorithm_code = config[algorithm]['algorithm_code']
            return {"learning_neighbors": learning_neighbors, "algorithm_code": algorithm_code}
        else:
            return {"learning_neighbors": learning_neighbors}


    elif algorithm == "mrnn":
        seq_len = int(config[algorithm]['seq_len'])
        epochs = int(config[algorithm]['epochs'])
        batch_size = int(config[algorithm]['batch_size'])
        sliding_windows = int(config[algorithm]['sliding_windows'])
        hidden_layers = int(config[algorithm]['hidden_layers'])
        impute_weight = float(config[algorithm]['impute_weight'])
        num_workers = int(config[algorithm]['num_workers'])
        return {
            "seq_len": seq_len,
            "epochs": epochs,
            "batch_size": batch_size,
            "sliding_windows": sliding_windows,
            "hidden_layers": hidden_layers,
            "impute_weight": impute_weight,
            "num_workers": num_workers,
        }


    elif algorithm == "iterative_svd":
        rank = int(config[algorithm]['rank'])
        return {"rank": rank}


    elif algorithm == "grouse":
        max_rank = int(config[algorithm]['max_rank'])
        return {"max_rank": max_rank}


    elif algorithm == "dynammo":
        h = int(config[algorithm]['h'])
        max_iteration = int(config[algorithm]['max_iteration'])
        approximation = bool(config[algorithm]['approximation'])
        return {"h": h, "max_iteration": max_iteration, "approximation": approximation}


    elif algorithm == "rosl":
        rank = int(config[algorithm]['rank'])
        regularization = float(config[algorithm]['regularization'])
        return {"rank": rank, "regularization": regularization}

    elif algorithm == "soft_impute":
        max_rank = int(config[algorithm]['max_rank'])
        return {"max_rank": max_rank}


    elif algorithm == "spirit":
        k = int(config[algorithm]['k'])
        w = int(config[algorithm]['w'])
        lvalue = float(config[algorithm]['lvalue'])
        return {"k": k, "w": w, "lvalue": lvalue}


    elif algorithm == "svt":
        tau = float(config[algorithm]['tau'])
        return {"tau": tau}


    elif algorithm == "tkcm":
        rank = int(config[algorithm]['rank'])
        return {"rank": rank}


    elif algorithm == "deep_mvi":
        max_epoch = int(config[algorithm]['max_epoch'])
        patience = int(config[algorithm]['patience'])
        lr = float(config[algorithm]['lr'])
        batch_size = int(config[algorithm]['batch_size'])
        return {
            "max_epoch": max_epoch,
            "patience": patience,
            "lr": lr,
            "batch_size": batch_size,
        }


    elif algorithm == "brits":
        model = str(config[algorithm]['model'])
        seq_len = int(config[algorithm]['seq_len'])
        epochs = int(config[algorithm]['epochs'])
        batch_size = int(config[algorithm]['batch_size'])
        sliding_windows = int(config[algorithm]['sliding_windows'])
        hidden_layers = int(config[algorithm]['hidden_layers'])
        impute_weight = float(config[algorithm]['impute_weight'])
        num_workers = int(config[algorithm]['num_workers'])
        return {
            "model": model,
            "seq_len": seq_len,
            "epochs": epochs,
            "batch_size": batch_size,
            "sliding_windows": sliding_windows,
            "hidden_layers": hidden_layers,
            "impute_weight": impute_weight,
            "num_workers": num_workers,
        }


    elif algorithm == "mpin":
        window = int(config[algorithm]['window'])
        incre_mode = str(config[algorithm]['incre_mode'])
        base = str(config[algorithm]['base'])
        epochs = int(config[algorithm]['epochs'])
        num_of_iteration = int(config[algorithm]['num_of_iteration'])
        k = int(config[algorithm]['k'])
        return {
            "window": window,
            "incre_mode": incre_mode,
            "base": base,
            "epochs": epochs,
            "num_of_iteration": num_of_iteration,
            "k": k,
        }


    elif algorithm == "knn" or algorithm == "knn_impute":
        k = int(config[algorithm]['k'])
        weights = str(config[algorithm]['weights'])
        return {"k": k, "weights": weights}


    elif algorithm == "interpolation":
        method = str(config[algorithm]['method'])
        poly_order = int(config[algorithm]['poly_order'])
        return {"method": method, "poly_order": poly_order}


    elif algorithm == "trmf":
        lags = list(config[algorithm]['lags'])
        K = int(config[algorithm]['K'])
        lambda_f = float(config[algorithm]['lambda_f'])
        lambda_x = float(config[algorithm]['lambda_x'])
        lambda_w = float(config[algorithm]['lambda_w'])
        eta = float(config[algorithm]['eta'])
        alpha = float(config[algorithm]['alpha'])
        max_iter = int(config[algorithm]['max_iter'])
        return {
            "lags": lags,
            "K": K,
            "lambda_f": lambda_f,
            "lambda_x": lambda_x,
            "lambda_w": lambda_w,
            "eta": eta,
            "alpha": alpha,
            "max_iter": max_iter,
        }


    elif algorithm == "mice":
        max_iter = int(config[algorithm]['max_iter'])
        tol = float(config[algorithm]['tol'])
        initial_strategy = str(config[algorithm]['initial_strategy'])
        seed = int(config[algorithm]['seed'])
        return {
            "max_iter": max_iter,
            "tol": tol,
            "initial_strategy": initial_strategy,
            "seed": seed,
        }

    elif algorithm == "miss_forest":
        n_estimators = int(config[algorithm]['n_estimators'])
        max_iter = int(config[algorithm]['max_iter'])
        max_features = str(config[algorithm]['max_features'])
        seed = int(config[algorithm]['seed'])
        return {
            "n_estimators": n_estimators,
            "max_iter": max_iter,
            "max_features": max_features,
            "seed": seed,
        }


    elif algorithm == "xgboost":
        n_estimators = int(config[algorithm]['n_estimators'])
        seed = int(config[algorithm]['seed'])
        return {"n_estimators": n_estimators, "seed": seed}


    elif algorithm == "miss_net":
        n_components = int(config[algorithm]['n_components'])
        alpha = float(config[algorithm]['alpha'])
        beta = float(config[algorithm]['beta'])
        n_cl = int(config[algorithm]['n_cl'])
        max_iter = int(config[algorithm]['max_iter'])
        tol = float(config[algorithm]['tol'])
        random_init = bool(config[algorithm]['random_init'])
        return {
            "n_components": n_components,
            "alpha": alpha,
            "beta": beta,
            "n_cl": n_cl,
            "max_iter": max_iter,
            "tol": tol,
            "random_init": random_init,
        }


    elif algorithm == "gain":
        batch_size = int(config[algorithm]['batch_size'])
        epochs = int(config[algorithm]['epochs'])
        alpha = int(config[algorithm]['alpha'])
        hint_rate = float(config[algorithm]['hint_rate'])
        return {
            "batch_size": batch_size,
            "epochs": epochs,
            "alpha": alpha,
            "hint_rate": hint_rate,
        }


    elif algorithm == "grin":
        seq_len = int(config[algorithm]['seq_len'])
        sim_type = str(config[algorithm]['sim_type'])
        epochs = int(config[algorithm]['epochs'])
        batch_size = int(config[algorithm]['batch_size'])
        sliding_windows = int(config[algorithm]['sliding_windows'])
        adj_threshold = float(config[algorithm]['adj_threshold'])
        patience = int(config[algorithm]['patience'])
        num_workers = int(config[algorithm]['num_workers'])
        return {
            "seq_len": seq_len,
            "sim_type": sim_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "sliding_windows": sliding_windows,
            "adj_threshold": adj_threshold,
            "patience": patience,
            "num_workers": num_workers,
        }


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
        return {
            "K_trend": K_trend,
            "K_season": K_season,
            "n_season": n_season,
            "K_bias": K_bias,
            "time_scale": time_scale,
            "a0": a0,
            "b0": b0,
            "v": v,
            "num_fold": num_fold,
        }

    elif algorithm == "hkmf_t":
        tags = config[algorithm]['tags']
        seq_len = int(config[algorithm]['seq_len'])
        blackouts_begin = int(config[algorithm]['blackouts_begin'])
        blackouts_end = int(config[algorithm]['blackouts_end'])
        epochs = int(config[algorithm]['epochs'])
        return {
            "tags": tags,
            "seq_len": seq_len,
            "blackouts_begin": blackouts_begin,
            "blackouts_end": blackouts_end,
            "epochs": epochs,
        }


    elif algorithm == "nuwats":
        seq_len = int(config[algorithm]['seq_len'])
        batch_size = int(config[algorithm]['batch_size'])
        epochs = int(config[algorithm]['epochs'])
        gpt_layers = int(config[algorithm]['gpt_layers'])
        num_workers = int(config[algorithm]['num_workers'])
        seed = int(config[algorithm]['seed'])
        return {
            "seq_len": seq_len,
            "batch_size": batch_size,
            "epochs": epochs,
            "gpt_layers": gpt_layers,
            "num_workers": num_workers,
            "seed": seed,
        }

    elif algorithm == "gpt4ts":
        seq_len = int(config[algorithm]['seq_len'])
        batch_size = int(config[algorithm]['batch_size'])
        epochs = int(config[algorithm]['epochs'])
        gpt_layers = int(config[algorithm]['gpt_layers'])
        num_workers = int(config[algorithm]['num_workers'])
        seed = int(config[algorithm]['seed'])
        return {
            "seq_len": seq_len,
            "batch_size": batch_size,
            "epochs": epochs,
            "gpt_layers": gpt_layers,
            "num_workers": num_workers,
            "seed": seed,
        }

    elif algorithm == "pristi":
        seq_len = int(config[algorithm]['seq_len'])
        batch_size = int(config[algorithm]['batch_size'])
        epochs = int(config[algorithm]['epochs'])
        sliding_windows = int(config[algorithm]['sliding_windows'])
        target_strategy = str(config[algorithm]['target_strategy'])
        nsamples = int(config[algorithm]['nsamples'])
        beta = config[algorithm]['beta']
        num_workers = int(config[algorithm]['num_workers'])
        return {
            "seq_len": seq_len,
            "batch_size": batch_size,
            "epochs": epochs,
            "sliding_windows": sliding_windows,
            "target_strategy": target_strategy,
            "nsamples": nsamples,
            "beta": beta,
            "num_workers": num_workers,
        }


    elif algorithm == "csdi":
        seq_len = int(config[algorithm]['seq_len'])
        batch_size = int(config[algorithm]['batch_size'])
        epochs = int(config[algorithm]['epochs'])
        sliding_windows = int(config[algorithm]['sliding_windows'])
        target_strategy = str(config[algorithm]['target_strategy'])
        nsamples = int(config[algorithm]['nsamples'])
        beta = config[algorithm]['beta']
        num_workers = int(config[algorithm]['num_workers'])
        return {
            "seq_len": seq_len,
            "batch_size": batch_size,
            "epochs": epochs,
            "sliding_windows": sliding_windows,
            "target_strategy": target_strategy,
            "nsamples": nsamples,
            "beta": beta,
            "num_workers": num_workers,
        }

    elif algorithm == "timesnet":
        seq_len = int(config[algorithm]['seq_len'])
        batch_size = int(config[algorithm]['batch_size'])
        top_k = int(config[algorithm]['top_k'])
        epochs = int(config[algorithm]['epochs'])
        gpt_layers = int(config[algorithm]['gpt_layers'])
        num_workers = int(config[algorithm]['num_workers'])
        seed = int(config[algorithm]['seed'])
        return {
            "seq_len": seq_len,
            "batch_size": batch_size,
            "top_k": top_k,
            "epochs": epochs,
            "gpt_layers": gpt_layers,
            "num_workers": num_workers,
            "seed": seed,
        }

    elif algorithm == "bit_graph":
        seq_len = int(config[algorithm]['seq_len'])
        sliding_windows = int(config[algorithm]['sliding_windows'])
        kernel_size = int(config[algorithm]['kernel_size'])
        kernel_set = config[algorithm]['kernel_set']
        epochs = int(config[algorithm]['epochs'])
        batch_size = int(config[algorithm]['batch_size'])
        subgraph_size = int(config[algorithm]['subgraph_size'])
        num_workers = int(config[algorithm]['num_workers'])
        return {
            "seq_len": seq_len,
            "sliding_windows": sliding_windows,
            "kernel_size": kernel_size,
            "kernel_set": kernel_set,
            "epochs": epochs,
            "batch_size": batch_size,
            "subgraph_size": subgraph_size,
            "num_workers": num_workers,
        }


    elif algorithm == "saits":
        seq_len = int(config[algorithm]['seq_len'])
        batch_size = int(config[algorithm]['batch_size'])
        epochs = int(config[algorithm]['epochs'])
        sliding_windows = int(config[algorithm]['sliding_windows'])
        n_head = int(config[algorithm]['n_head'])
        num_workers = int(config[algorithm]['num_workers'])
        return {
            "seq_len": seq_len,
            "batch_size": batch_size,
            "epochs": epochs,
            "sliding_windows": sliding_windows,
            "n_head": n_head,
            "num_workers": num_workers,
        }

    elif algorithm == "moment":
        model = str(config[algorithm]['model'])
        model_size = str(config[algorithm]['model_size'])
        use_mean = bool(config[algorithm]['use_mean'])
        seq_len = int(config[algorithm]['seq_len'])
        sliding_windows = int(config[algorithm]['sliding_windows'])
        return {
            "model": model,
            "model_size": model_size,
            "use_mean": use_mean,
            "seq_len": seq_len,
            "sliding_windows": sliding_windows,
        }


    # OPTI ================================================================

    elif algorithm.lower() in ["test-greedy", "greedy"]:
        n_calls = int(config["greedy"]['n_calls'])
        metrics = config["greedy"]['metrics']
        return (n_calls, [metrics])
    elif algorithm.lower() in ["test-bayesian", "bayesian", "bo", "bayesopt"]:
        n_calls = int(config['bayesian']['n_calls'])
        n_random_starts = int(config['bayesian']['n_random_starts'])
        acq_func = str(config['bayesian']['acq_func'])
        metrics = config['bayesian']['metrics']
        return (n_calls, n_random_starts, acq_func, [metrics])
    elif algorithm.lower() in ['test-pso', 'pso', "particle_swarm"]:
        n_particles = int(config['pso']['n_particles'])
        c1 = float(config['pso']['c1'])
        c2 = float(config['pso']['c2'])
        w = float(config['pso']['w'])
        iterations = int(config['pso']['iterations'])
        n_processes = int(config['pso']['n_processes'])
        metrics = config['pso']['metrics']
        return (n_particles, c1, c2, w, iterations, n_processes, [metrics])
    elif algorithm.lower() in  ['test-sh', 'sh', "successive_halving"]:
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
        alpha = float(config[algorithm]['alpha'])
        return {"seasonality_mode": seasonality_mode, "n_changepoints": n_changepoints, "alpha":alpha}
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
    elif algorithm == "forecaster-ltsf":
        seq_len = int(config[algorithm]['seq_len'])
        pred_len = float(config[algorithm]['pred_len'])
        return {"seq_len": seq_len, "pred_len": pred_len}
    elif algorithm == "colors":
        colors = config[algorithm]['plot']
        return colors
    elif algorithm == "colors_blacks":
        colors = config[algorithm]['plot']
        return colors



    # new ones
    elif algorithm == "forecaster-dlinear":
        input_chunk_length = int(config[algorithm]['input_chunk_length'])
        output_chunk_length = int(config[algorithm]['output_chunk_length'])
        random_state = int(config[algorithm]['random_state'])
        n_epochs = int(config[algorithm]['n_epochs'])
        pl_trainer_kwargs = str(config[algorithm]['pl_trainer_kwargs'])
        if pl_trainer_kwargs == "cpu":
            drive = {"accelerator": pl_trainer_kwargs}
        else:
            drive = {"accelerator": pl_trainer_kwargs, "devices": [0]}
        return {"input_chunk_length": input_chunk_length, "output_chunk_length": output_chunk_length, "random_state": random_state, "n_epochs": n_epochs, "pl_trainer_kwargs": drive,}

    elif algorithm == "forecaster-dlinear":
        input_chunk_length = int(config[algorithm]['input_chunk_length'])
        output_chunk_length = int(config[algorithm]['output_chunk_length'])
        random_state = int(config[algorithm]['random_state'])
        n_epochs = int(config[algorithm]['n_epochs'])
        pl_trainer_kwargs = str(config[algorithm]['pl_trainer_kwargs'])
        if pl_trainer_kwargs == "cpu":
            drive = {"accelerator": pl_trainer_kwargs}
        else:
            drive = {"accelerator": pl_trainer_kwargs, "devices": [0]}
        return {"input_chunk_length": input_chunk_length, "output_chunk_length": output_chunk_length, "random_state": random_state, "n_epochs": n_epochs, "pl_trainer_kwargs": drive,}

    elif algorithm == "forecaster-nlinear":
        input_chunk_length = int(config[algorithm]['input_chunk_length'])
        output_chunk_length = int(config[algorithm]['output_chunk_length'])
        normalize = bool(config[algorithm]['output_chunk_length'])
        random_state = int(config[algorithm]['random_state'])
        n_epochs = int(config[algorithm]['n_epochs'])
        pl_trainer_kwargs = str(config[algorithm]['pl_trainer_kwargs'])
        if pl_trainer_kwargs == "cpu":
            drive = {"accelerator": pl_trainer_kwargs}
        else:
            drive = {"accelerator": pl_trainer_kwargs, "devices": [0]}
        return {"input_chunk_length": input_chunk_length, "output_chunk_length": output_chunk_length, "normalize": normalize, "random_state": random_state, "n_epochs": n_epochs, "pl_trainer_kwargs": drive, }

    elif algorithm == "forecaster-chronos":
        input_chunk_length = int(config[algorithm]['input_chunk_length'])
        output_chunk_length = int(config[algorithm]['output_chunk_length'])
        hub_model_name = str(config[algorithm].get('hub_model_name', 'amazon/chronos-2'))
        random_state = int(config[algorithm]['random_state'])
        n_epochs = int(config[algorithm]['n_epochs'])
        pl_trainer_kwargs = str(config[algorithm]['pl_trainer_kwargs'])
        if pl_trainer_kwargs == "cpu":
            drive = {"accelerator": "cpu"}
        else:
            drive = {"accelerator": pl_trainer_kwargs, "devices": [0]}
        return {"input_chunk_length": input_chunk_length, "output_chunk_length": output_chunk_length, "hub_model_name": hub_model_name, "random_state": random_state, "n_epochs": n_epochs, "pl_trainer_kwargs": drive, }


    elif algorithm == "forecaster-patchtst":
        from datetime import datetime
        fit_strategy = str(config[algorithm].get("fit_strategy", "full"))
        validation_split = float(config[algorithm].get("validation_split", 0.2))
        patch_length = int(config[algorithm]["patch_length"])
        context_length = int(config[algorithm]["context_length"])
        patch_stride = int(config[algorithm]["patch_stride"])
        d_model = int(config[algorithm]["d_model"])
        num_attention_heads = int(config[algorithm]["num_attention_heads"])
        ffn_dim = int(config[algorithm]["ffn_dim"])
        head_dropout = float(config[algorithm]["head_dropout"])
        pred_len = int(config[algorithm]["pred_len"])
        model_config = {"patch_length": patch_length, "context_length": context_length, "patch_stride": patch_stride, "d_model": d_model, "num_attention_heads": num_attention_heads, "ffn_dim": ffn_dim, "head_dropout": head_dropout, "prediction_length": pred_len, }
        date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_args = {
            "output_dir": str(config[algorithm].get(
                "output_dir",
                f"./patchtst_output_{date_tag}"
            )),
            "num_train_epochs": int(config[algorithm].get("num_train_epochs", 10)),
            "per_device_train_batch_size": int(config[algorithm].get("per_device_train_batch_size", 16)),
            "learning_rate": float(config[algorithm].get("learning_rate", 1e-4)),
            "report_to": "none",
        }
        return {"fit_strategy": fit_strategy, "validation_split": validation_split, "config": model_config, "training_args": training_args,}


    elif algorithm == "forecaster-moment":
        pretrained_model_name_or_path = str(config[algorithm].get("pretrained_model_name_or_path", "AutonLab/MOMENT-1-large"))
        dropout = float(config[algorithm].get("dropout", 0.1))
        head_dropout = float(config[algorithm].get("head_dropout", 0.1))
        seq_len = int(config[algorithm].get("seq_len", 512))
        batch_size = int(config[algorithm].get("batch_size", 32))
        eval_batch_size = int(config[algorithm].get("eval_batch_size", 32))
        epochs = int(config[algorithm].get("epochs", 1))
        max_lr = float(config[algorithm].get("max_lr", 1e-4))
        train_val_split = float(config[algorithm].get("train_val_split", 0.1))
        pl_trainer_kwargs = str(config[algorithm]['pl_trainer_kwargs'])
        if pl_trainer_kwargs == "cpu":
            drive = "cpu"#{"accelerator": "cpu"}
        else:
            drive = {"accelerator": pl_trainer_kwargs, "devices": [0]}

        return {"pretrained_model_name_or_path": pretrained_model_name_or_path, "dropout": dropout, "head_dropout": head_dropout, "seq_len": seq_len, "batch_size": batch_size, "eval_batch_size": eval_batch_size, "epochs": epochs, "max_lr": max_lr, "device": drive, "train_val_split": train_val_split,}

    # ====


    elif algorithm == "classifier-forest":
        min_interval = int(config[algorithm]["min_interval"])
        n_estimators = int(config[algorithm]["n_estimators"])
        n_jobs = int(config[algorithm]["n_jobs"])
        random_state = int(config[algorithm]["random_state"])
        return {"min_interval": min_interval, "n_estimators": n_estimators, "n_jobs": n_jobs, "random_state": random_state, }

    elif algorithm == "classifier-muse":
        anova = bool(config[algorithm]["anova"])
        window_inc = int(config[algorithm]["window_inc"])
        alphabet_size = int(config[algorithm]["alphabet_size"])
        n_jobs = int(config[algorithm]["n_jobs"])
        random_state = int(config[algorithm]["random_state"])
        return {"anova": anova, "window_inc": window_inc, "alphabet_size": alphabet_size, "n_jobs": n_jobs, "random_state": random_state, }

    elif algorithm == "classifier-weasel":
        anova = bool(config[algorithm]["anova"])
        p_threshold = float(config[algorithm]["p_threshold"])
        alphabet_size = int(config[algorithm]["alphabet_size"])
        window_inc = int(config[algorithm]["window_inc"])
        n_jobs = int(config[algorithm]["n_jobs"])
        random_state = int(config[algorithm]["random_state"])
        return {"anova": anova, "window_inc": window_inc, "p_threshold":p_threshold, "alphabet_size":alphabet_size, "n_jobs": n_jobs, "random_state": random_state, }

    elif algorithm == "classifier-itde":
        window_size = int(config[algorithm]["window_size"])
        word_length = int(config[algorithm]["word_length"])
        max_dims = int(config[algorithm]["max_dims"])
        dim_threshold = float(config[algorithm]["dim_threshold"])
        n_jobs = int(config[algorithm]["n_jobs"])
        random_state = int(config[algorithm]["random_state"])
        return {"window_size": window_size, "word_length": word_length, "max_dims": max_dims, "n_jobs": n_jobs, "random_state": random_state, }

    elif algorithm == "classifier-tde":
        n_parameter_samples = int(config[algorithm]["n_parameter_samples"])
        max_dims = int(config[algorithm]["max_dims"])
        n_jobs = int(config[algorithm]["n_jobs"])
        random_state = int(config[algorithm]["random_state"])
        return {"n_parameter_samples": n_parameter_samples, "max_dims": max_dims, "n_jobs": n_jobs, "random_state": random_state, }

    elif algorithm == "classifier-cboss":
        n_parameter_samples = int(config[algorithm]["n_parameter_samples"])
        max_ensemble_size = int(config[algorithm]["max_ensemble_size"])
        min_window = int(config[algorithm]["min_window"])
        n_jobs = int(config[algorithm]["n_jobs"])
        random_state = int(config[algorithm]["random_state"])
        return {"n_parameter_samples": n_parameter_samples, "max_ensemble_size":max_ensemble_size, "min_window": min_window, "n_jobs": n_jobs, "random_state": random_state, }

    elif algorithm == "classifier-knn":
        n_neighbors = int(config[algorithm]["n_neighbors"])
        distance = str(config[algorithm]["distance"])
        leaf_size = int(config[algorithm]["leaf_size"])
        n_jobs = int(config[algorithm]["n_jobs"])
        return {"n_neighbors": n_neighbors, "distance": distance, "leaf_size":leaf_size, "n_jobs": n_jobs, }

    elif algorithm == "classifier-proxforest":
        random_state = int(config[algorithm]["random_state"])
        n_estimators = int(config[algorithm]["n_estimators"])
        n_jobs = int(config[algorithm]["n_jobs"])
        n_stump_evaluations = int(config[algorithm]["n_stump_evaluations"])
        return {"random_state": random_state, "n_estimators": n_estimators, "n_jobs": n_jobs, "n_stump_evaluations": n_stump_evaluations, }

    elif algorithm == "classifier-proxtree":
        random_state = int(config[algorithm]["random_state"])
        max_depth = float(config[algorithm]["max_depth"])
        n_jobs = int(config[algorithm]["n_jobs"])
        n_stump_evaluations = int(config[algorithm]["n_stump_evaluations"])
        return {"random_state": random_state, "max_depth": max_depth, "n_jobs": n_jobs, "n_stump_evaluations": n_stump_evaluations, }

    elif algorithm == "classifier-proxstump":
        random_state = int(config[algorithm]["random_state"])
        verbosity = int(config[algorithm]["verbosity"])
        n_jobs = int(config[algorithm]["n_jobs"])
        return {"random_state": random_state, "verbosity": verbosity, "n_jobs": n_jobs, }

    elif algorithm == "classifier-shapedtw":
        n_neighbors = int(config[algorithm]["n_neighbors"])
        subsequence_length = int(config[algorithm]["subsequence_length"])
        return {"n_neighbors": n_neighbors, "subsequence_length": subsequence_length, }

    elif algorithm == "classifier-tsf":
        min_interval = int(config[algorithm]["min_interval"])
        n_estimators = int(config[algorithm]["n_estimators"])
        n_jobs = int(config[algorithm]["n_jobs"])
        random_state = int(config[algorithm]["random_state"])
        return {"min_interval": min_interval, "n_estimators": n_estimators, "n_jobs": n_jobs, "random_state": random_state, }

    elif algorithm == "classifier-cif":
        n_estimators = int(config[algorithm]["n_estimators"])
        att_subsample_size = int(config[algorithm]["att_subsample_size"])
        min_interval = int(config[algorithm]["min_interval"])
        n_jobs = int(config[algorithm]["n_jobs"])
        random_state = int(config[algorithm]["random_state"])
        return {"n_estimators": n_estimators, "att_subsample_size":att_subsample_size, "min_interval": min_interval, "n_jobs": n_jobs, "random_state": random_state, }

    elif algorithm == "classifier-stc":
        n_shapelet_samples = int(config[algorithm]["n_shapelet_samples"])
        n_jobs = int(config[algorithm]["n_jobs"])
        batch_size = int(config[algorithm]["batch_size"])
        random_state = int(config[algorithm]["random_state"])
        return {"n_shapelet_samples": n_shapelet_samples, "n_jobs": n_jobs, "batch_size": batch_size, "random_state": random_state, }

    elif algorithm == "classifier-lstm":
        n_epochs = int(config[algorithm]["n_epochs"])
        lstm_size = int(config[algorithm]["lstm_size"])
        random_state = int(config[algorithm]["random_state"])
        verbose = int(config[algorithm]["verbose"])
        return {"n_epochs": n_epochs, "lstm_size":lstm_size, "random_state": random_state, "verbose": verbose, }

    elif algorithm == "classifier-cnn":
        n_epochs = int(config[algorithm]["n_epochs"])
        n_conv_layers = int(config[algorithm]["n_conv_layers"])
        random_state = int(config[algorithm]["random_state"])
        verbose = bool(config[algorithm]["verbose"])
        return {"n_epochs": n_epochs, "n_conv_layers":n_conv_layers, "random_state": random_state, "verbose": verbose, }

    elif algorithm == "classifier-svc":
        C = float(config[algorithm]["C"])
        tol = float(config[algorithm]["tol"])
        max_iter = int(config[algorithm]["max_iter"])
        cache_size = int(config[algorithm]["cache_size"])
        random_state = int(config[algorithm]["random_state"])
        return {"C": C, "tol":tol, "max_iter": max_iter, "cache_size":cache_size, "random_state": random_state, }

    elif algorithm == "classifier-arsenal":
        num_kernels = int(config[algorithm]["num_kernels"])
        n_estimators = int(config[algorithm]["n_estimators"])
        n_jobs = int(config[algorithm]["n_jobs"])
        random_state = int(config[algorithm]["random_state"])
        return {"num_kernels": num_kernels, "n_estimators": n_estimators, "n_jobs": n_jobs, "random_state": random_state, }

    elif algorithm == "classifier-rocket":
        num_kernels = int(config[algorithm]["num_kernels"])
        n_jobs = int(config[algorithm]["n_jobs"])
        random_state = int(config[algorithm]["random_state"])
        return {"num_kernels": num_kernels, "n_jobs": n_jobs, "random_state": random_state, }

    elif algorithm == "classifier-catch22":
        estimator = int(config[algorithm]["estimator"])
        n_jobs = int(config[algorithm]["n_jobs"])
        random_state = int(config[algorithm]["random_state"])
        return {"estimator": estimator, "n_jobs": n_jobs, "random_state": random_state, }

    elif algorithm == "classifier-mpc":
        subsequence_length = int(config[algorithm]["subsequence_length"])
        n_jobs = int(config[algorithm]["n_jobs"])
        random_state = int(config[algorithm]["random_state"])
        return {"subsequence_length": subsequence_length, "n_jobs": n_jobs, "random_state": random_state, }

    elif algorithm == "classifier-signature":
        window_name = str(config[algorithm]["window_name"])
        window_depth = int(config[algorithm]["window_depth"])
        depth = int(config[algorithm]["depth"])
        random_state = int(config[algorithm]["random_state"])
        return {"window_name":window_name, "window_depth":window_depth, "depth": depth, "random_state": random_state, }

    elif algorithm == "classifier-tsfresh":
        relevant_feature_extractor = bool(config[algorithm]["relevant_feature_extractor"])
        n_jobs = int(config[algorithm]["n_jobs"])
        random_state = int(config[algorithm]["random_state"])
        return {"relevant_feature_extractor": relevant_feature_extractor, "n_jobs": n_jobs, "random_state": random_state, }



    elif algorithm == "other":
            return config

        # Your own default parameters #contributing-load-params
        #
        #elif algorithm == "your_algo_name":
        #    param1 = int(config[algorithm]['param_1'])
        #    param2 = config[algorithm]['param_2']
        #    param3 = float(config[algorithm]['param_3'])
        #    return {"param_1": param1,"param_2": param2, "param_3": param3}

    elif algorithm == "mean_impute":
        logic = str(config[algorithm]["logic"])
        return {"logic": logic }

    else:
        print("(SYS) Default/Optimal config not found for this algorithm")
        return None


def config_impute_algorithm(incomp_data, algorithm, task="default", verbose=True):
    """
    Initialize and configure the selected imputation algorithm for the specified task.

    Parameters
    ----------
    incomp_data : TimeSeries
        TimeSeries object containing dataset.

    algorithm : str
        Name of algorithm

    task: string, optional
        Task of the model (default is "default"). It will load the default parameters for the specific task.
        "default" for regular imputation / "classification" for classification. / "forecasting" for forecasting.

    verbose : bool, optional
        Whether to display execution information (default is True).

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
    elif alg == "cdrec2":
        imputer = Imputation.MatrixCompletion.CDRec2(incomp_data)
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
    elif alg == "moment":
        imputer = Imputation.LLMs.Moment(incomp_data)

    # your own implementation #contributing-add-toolkit
    #
    #elif alg == "your_algo_name":
    #    imputer = Imputation.MyFamily.NewAlg(incomp_data)
    else:
        raise ValueError(f"(IMP) Algorithm '{algorithm}' not recognized, please choose your algorithm from this list:\n\t{TimeSeries().algorithms}")
        imputer = None

    if imputer is not None:
        imputer.verbose = verbose
        imputer.task = task

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
    elif algorithm == "moment" or algorithm == "Moment":
        params_to_save = {
            "model": str(optimal_params[0]),
            "model_size": str(optimal_params[1]),
            "use_mean": bool(optimal_params[2]),
            "seq_len": int(optimal_params[3]),
            "sliding_windows": int(optimal_params[4])
        }

    # Your own optimal save parameters #contributing-optimization-save
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


def make_boring(X_train, X_test, y_train):
    # Step 1: transform test set from string/object to int and store a dictionary
    myDict = {}
    newTrain = []
    for i in range(0, len(y_train)):
        classVal = y_train[i]
        keys = [k for k, v in myDict.items() if v == classVal]
        if len(keys) == 0:
            keyVal = len(myDict)
            myDict[keyVal] = y_train[i]
        else:
            keyVal = keys[0]
        # endif
        newTrain.append(keyVal)
    # end for
    y_train = np.array(newTrain)

    # Step 2: transform train & test sets from fancy pandas indexed table to boring list of lists
    #X_train = X_train.to_numpy()
    #X_test = X_test.to_numpy()

    series = len(X_train)
    seriesTest = len(X_test)
    seriesLen = len(X_train[0])  # instead of len(X_train[0][0])

    newTrain = np.array([X_train[j][0][i] for j in range(0, series) for i in range(0, seriesLen)])
    newTest = np.array([X_test[j][0][i] for j in range(0, seriesTest) for i in range(0, seriesLen)])

    X_train = newTrain.reshape(series, seriesLen)
    X_test = newTest.reshape(seriesTest, seriesLen)

    return [X_train, X_test, y_train, myDict]


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
    Configure and apply a contamination pattern to a time-series dataset.

    Parameters
    ----------
    ts : TimeSeries or array-like
        Input time-series data. If a ``TimeSeries`` object is provided, its
        ``data`` attribute is used as the input matrix. Otherwise, ``ts`` is
        treated directly as the input data matrix.

    pattern : str
        Contamination pattern to apply. Supported patterns and aliases include:

        - ``"mcar"`` or ``"missing_completely_at_random"``
        - ``"mp"``, ``"missingpercentage"``, or ``"aligned"``
        - ``"ps"``, ``"percentageshift"``, ``"scattered"``, or ``"scatter"``
        - ``"disjoint"``
        - ``"overlap"``
        - ``"gaussian"``
        - ``"distribution"`` or ``"dist"``
        - ``"blackout"``
        - ``"alignedseries"``
        - ``"alignedtimestamps"``

        Underscores and hyphens are ignored when matching most pattern names.

    dataset_rate : float, default=0.4
        Fraction of the dataset affected by contamination. Its exact
        interpretation depends on the selected contamination pattern.

    series_rate : float, default=0.4
        Fraction of values or timestamps contaminated within each selected
        time series. Its exact interpretation depends on the selected pattern.

    block_size : int, default=10
        Size of consecutive missing-value blocks for MCAR contamination.

    offset : float, default=0.1
        Fraction of the beginning of each series protected from contamination.

    seed : bool or int, default=True
        Seed configuration used by stochastic contamination patterns to obtain
        reproducible results.

    limit : int, default=1
        Maximum overlap-related constraint used by the ``"overlap"`` pattern.

    shift : float, default=0.05
        Shift between missing blocks used by the ``"overlap"`` pattern.

    std_dev : float, default=0.5
        Standard deviation used by the ``"gaussian"`` contamination pattern.

    explainer : bool, default=False
        Whether to enable additional contamination information intended for
        explanation or analysis. Used only by patterns supporting this option.

    probabilities : sequence of float, optional
        Probability distribution passed to the ``"distribution"`` contamination
        pattern.

    logic_by_series : bool, default=True
        Whether contamination logic is applied independently at the series
        level when supported by the selected pattern.

    verbose : bool, default=True
        Whether to display contamination-related information during execution.

    Returns
    -------
    numpy array
        Contaminated time-series data returned by the selected ``GenGap``
        contamination method.

    Raises
    ------
    ValueError
        If ``pattern`` does not correspond to a supported contamination
        pattern.
    """


    from imputegap.recovery.contamination import GenGap
    from imputegap.recovery.manager import TimeSeries

    pattern_low = pattern.lower()
    ptn = pattern_low.replace('_', '').replace('-', '')

    if isinstance(ts, TimeSeries):
        matrix = ts.data
    else:
        matrix = ts

    if ptn == "mcar" or ptn == "missing_completely_at_random":
        incomp_data = GenGap.mcar(input_data=matrix, rate_dataset=dataset_rate, rate_series=series_rate, block_size=block_size, offset=offset, seed=seed, explainer=explainer, logic_by_series=logic_by_series, verbose=verbose)
    elif ptn == "mp" or ptn == "missingpercentage" or ptn == "aligned":
        incomp_data = GenGap.aligned(input_data=matrix, rate_dataset=dataset_rate, rate_series=series_rate, offset=offset, explainer=explainer, logic_by_series=logic_by_series, verbose=verbose)
    elif ptn == "ps" or ptn == "percentageshift" or ptn == "scattered" or ptn == "scatter":
        incomp_data = GenGap.scattered(input_data=matrix, rate_dataset=dataset_rate, rate_series=series_rate, offset=offset, seed=seed, explainer=explainer, logic_by_series=logic_by_series, verbose=verbose)
    elif ptn == "disjoint":
        incomp_data = GenGap.disjoint(input_data=matrix, rate_series=dataset_rate, limit=1, offset=offset, logic_by_series=logic_by_series, verbose=verbose)
    elif ptn == "overlap":
        incomp_data = GenGap.overlap(input_data=matrix, rate_series=dataset_rate, limit=limit, shift=shift, offset=offset, logic_by_series=logic_by_series, verbose=verbose)
    elif ptn == "gaussian":
        incomp_data = GenGap.gaussian(input_data=matrix, rate_dataset=dataset_rate, rate_series=series_rate, std_dev=std_dev, offset=offset, seed=seed, explainer=explainer, logic_by_series=logic_by_series, verbose=verbose)
    elif ptn == "distribution" or pattern == "dist":
        incomp_data = GenGap.distribution(input_data=matrix, rate_dataset=dataset_rate, rate_series=series_rate, probabilities_list=probabilities, offset=offset, seed=seed, explainer=explainer, logic_by_series=logic_by_series, verbose=verbose)
    elif ptn == "blackout":
        incomp_data = GenGap.blackout(input_data=matrix, rate_series=dataset_rate, offset=offset, logic_by_series=logic_by_series, verbose=verbose)
    elif ptn == "alignedseries":
        incomp_data = GenGap.aligned(input_data=matrix, rate_series=series_rate, rate_dataset=dataset_rate, offset=offset, logic_by_series=logic_by_series, verbose=verbose)
    elif ptn == "alignedtimestamps":
        incomp_data = GenGap.aligned(input_data=matrix, rate_series=series_rate, rate_dataset=dataset_rate, offset=offset, logic_by_series=logic_by_series, verbose=verbose)
    else:
        raise ValueError(f"\n(CONT) Pattern '{pattern}' not recognized, please choose your algorithm on this list :\n\t{TimeSeries().patterns}\n")
        incomp_data = None

    return incomp_data




def config_classifier(model, params):
    """
    Configure and instantiate a time-series classifier for downstream analytics.

    The classifier is selected from its model name and initialized using the
    provided parameters. Model names are case-insensitive, and underscores and
    hyphens are ignored during model matching.

    Parameters
    ----------
    model : str
        Name of the classifier model to instantiate.

        Supported classifiers are:

        Dictionary-based:
            - ``"muse"``       : MUSE
            - ``"weasel"``     : WEASEL
            - ``"itde"``       : IndividualTDE
            - ``"tde"``        : TemporalDictionaryEnsemble
            - ``"cboss"``      : ContractableBOSS

        Distance-based:
            - ``"knn"``        : KNeighborsTimeSeriesClassifier
            - ``"proxforest"`` : ProximityForest
            - ``"proxtree"``   : ProximityTree
            - ``"proxstump"``  : ProximityStump
            - ``"shapedtw"``   : ShapeDTW

        Hybrid:
            - ``"hivecote"``   : HIVECOTEV1
            - ``"hivecote2"``  : HIVECOTEV2

        Interval-based:
            - ``"forest"``     : TimeSeriesForestClassifier
            - ``"tsf"``        : TimeSeriesForestClassifier
            - ``"cif"``        : CanonicalIntervalForest

        Shapelet-based:
            - ``"stc"``        : ShapeletTransformClassifier

        Deep-learning:
            - ``"lstm"``       : LSTMFCNClassifier
            - ``"cnn"``        : CNNClassifier

        Kernel-based:
            - ``"svc"``        : TimeSeriesSVC
            - ``"arsenal"``    : Arsenal
            - ``"rocket"``     : RocketClassifier

        Feature-based:
            - ``"catch22"``    : Catch22Classifier
            - ``"mpc"``        : MatrixProfileClassifier
            - ``"signature"``  : SignatureClassifier
            - ``"tsfresh"``    : TSFreshClassifier

    params : dict
        Dictionary containing the parameters used to initialize the selected
        classifier.

        For most classifiers, the parameters are passed directly to the
        constructor using ``**params``.

    Returns
    -------
    classifier : object
        Instantiated ``sktime`` classifier corresponding to ``model``, ready to
        be used for downstream time-series classification.

    Raises
    ------
    ValueError
        If ``model`` does not correspond to one of the supported classifiers.

    Notes
    -----
    Model-name normalization is performed as follows:

        1. Convert the model name to lowercase.
        2. Remove underscores (``_``).
        3. Remove hyphens (``-``).

    For example, ``"prox-forest"`` and ``"prox_forest"`` are both normalized to
    ``"proxforest"`` before model selection.
    """

    from imputegap.recovery.manager import TimeSeries

    model_low = model.lower()
    mdl = model_low.replace('_', '').replace('-', '')

    if mdl == "forest":
        from sktime.classification.interval_based import TimeSeriesForestClassifier
        classifier = TimeSeriesForestClassifier(**params)

    #
    # Dictionary based
    #
    elif mdl == "muse":
        from sktime.classification.dictionary_based import MUSE
        classifier = MUSE(**params)

    elif mdl == "weasel":  # UNIVAR
        from sktime.classification.dictionary_based import WEASEL
        classifier = WEASEL(**params)
    elif mdl == "itde":
        from sktime.classification.dictionary_based import IndividualTDE
        classifier = IndividualTDE(**params)
    elif mdl == "tde":
        from sktime.classification.dictionary_based import TemporalDictionaryEnsemble
        classifier = TemporalDictionaryEnsemble(**params)
    elif mdl == "cboss":
        from sktime.classification.dictionary_based import ContractableBOSS
        classifier = ContractableBOSS(**params)
    #
    # Distance based
    #
    elif mdl == "knn":
        from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
        classifier = KNeighborsTimeSeriesClassifier(**params)

    elif mdl == "proxforest":
        from sktime.classification.distance_based import ProximityForest
        classifier = ProximityForest(**params)
    elif mdl == "proxtree":
        from sktime.classification.distance_based import ProximityTree
        classifier = ProximityTree(**params)
    elif mdl == "proxstump":
        from sktime.classification.distance_based import ProximityStump
        classifier = ProximityStump(**params)
    elif mdl == "shapedtw":
        from sktime.classification.distance_based import ShapeDTW
        classifier = ShapeDTW(**params)
    #
    # Hybrid
    #
    elif mdl == "hivecote":
        from sktime.classification.hybrid import HIVECOTEV1
        classifier = HIVECOTEV1()
    elif mdl == "hivecote2":
        from sktime.classification.hybrid import HIVECOTEV2
        classifier = HIVECOTEV2()
    #
    # Interval based
    #
    elif mdl == "tsf":
        from sktime.classification.interval_based import TimeSeriesForestClassifier
        classifier = TimeSeriesForestClassifier(**params)
    elif mdl == "cif":
        import warnings
        warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning, module="sklearn.utils.deprecation",)
        from sktime.classification.interval_based import CanonicalIntervalForest
        classifier = CanonicalIntervalForest(**params)
    #
    # Shapelet based
    #
    elif mdl == "stc":
        from sktime.classification.shapelet_based import ShapeletTransformClassifier
        classifier = ShapeletTransformClassifier(**params)
    #
    # NN based
    #
    elif mdl == "lstm":
        from sktime.classification.deep_learning import LSTMFCNClassifier
        classifier = LSTMFCNClassifier(**params)

    elif mdl == "cnn":
        from sktime.classification.deep_learning.cnn import CNNClassifier
        classifier = CNNClassifier(**params)

    #
    # Kernel based
    #
    elif mdl == "svc":
        from sktime.classification.kernel_based import TimeSeriesSVC
        classifier = TimeSeriesSVC(**params)

    elif mdl == "arsenal":
        from sktime.classification.kernel_based import Arsenal
        classifier = Arsenal(**params)

    elif mdl == "rocket":
        from sktime.classification.kernel_based import RocketClassifier
        classifier = RocketClassifier(**params)

    #
    # Feature based
    #
    elif mdl == "catch22":
        from sktime.classification.feature_based import Catch22Classifier
        from sklearn.ensemble import RandomForestClassifier
        estimators, n_jobs, random_state = params.values()
        classifier = Catch22Classifier(estimator=RandomForestClassifier(n_estimators=estimators), n_jobs=n_jobs, random_state=random_state)

    elif mdl == "mpc":
        from sktime.classification.feature_based import MatrixProfileClassifier
        classifier = MatrixProfileClassifier(**params)

    elif mdl == "signature":
        from sktime.classification.feature_based import SignatureClassifier
        classifier = SignatureClassifier(**params)

    elif mdl == "tsfresh":
        from sktime.classification.feature_based import TSFreshClassifier
        classifier = TSFreshClassifier(**params)

    else:
        raise ValueError(f"\n(DOWN) Classifier model '{model}' not recognized, please choose your algorithm on this list :\n\t{list_of_classifiers()}\n")
        classifier = None

    return classifier


def config_forecaster(model, params, pred_len=None, s_exception=False):
    """
    Configure and instantiate a time-series forecaster for downstream analytics.

    The forecaster is selected from its model name and initialized using the
    provided parameters. Model names are case-insensitive, and underscores and
    hyphens are ignored during model matching.

    The forecasting horizon can optionally be propagated to model parameters
    supporting ``pred_len`` or ``config["prediction_length"]``.

    Parameters
    ----------
    model : str
        Name of the forecasting model to instantiate.

        Supported forecasters are:

        Statistical / classical:
            - ``"prophet"``      : Prophet
            - ``"expsmoothing"`` : ExponentialSmoothing
            - ``"hwadd"``        : ExponentialSmoothing
            - ``"arima"``        : AutoARIMA
            - ``"sf-arima"``     : StatsForecastAutoARIMA
            - ``"bats"``         : BATS
            - ``"ets"``          : AutoETS
            - ``"croston"``      : Croston
            - ``"theta"``        : ThetaForecaster
            - ``"unobs"``        : UnobservedComponents
            - ``"naive"``        : NaiveForecaster

        Machine-learning:
            - ``"xgboost"``      : XGBModel
            - ``"lightgbm"``     : LightGBMModel

        Deep-learning:
            - ``"nbeats"``       : NBEATSModel
            - ``"lstm"``         : RNNModel
            - ``"deepar"``       : RNNModel
            - ``"transformer"``  : TransformerModel
            - ``"dlinear"``      : DLinearModel
            - ``"nlinear"``      : NLinearModel
            - ``"ltsf"``         : LTSFLinearForecaster

        Foundation / pretrained:
            - ``"chronos"``      : Chronos2Model
            - ``"patchtst"``     : PatchTSTForecaster
            - ``"moment"``       : MomentFMForecaster

    params : dict
        Dictionary containing the parameters used to initialize the selected
        forecaster. Parameters are passed to the corresponding constructor using
        ``**params``.

        This dictionary may be modified in place when ``pred_len`` or
        ``s_exception`` requires a configuration adaptation.

    pred_len : int, optional
        Forecasting horizon used to adapt model parameters when supported.

        If ``params`` contains ``"pred_len"``, its value is replaced by
        ``pred_len``.

        If ``params`` contains a ``"config"`` dictionary with a
        ``"prediction_length"`` entry, that value is also replaced by
        ``pred_len``.

        If ``None``, no horizon adaptation is performed.

    s_exception : bool, default=False
        Whether to activate the ARIMA-specific configuration patch.

        When enabled for ``"arima"``, the following parameters are forced:

            - ``sp = 24``
            - ``max_p = 2``
            - ``max_q = 2``

    Returns
    -------
    forecaster : object
        Instantiated ``sktime`` or ``darts`` forecasting model corresponding to
        ``model``, ready to be used for downstream forecasting.

    Raises
    ------
    ValueError
        If ``model`` does not correspond to one of the supported forecasting
        models.

    Configuration adaptations
    -------------------------

    MODEL    CONDITION                    > PARAMETER              VALUE
    -----------------------------------------------------------------------
    ALL      pred_len is provided        > pred_len               pred_len
    ALL      prediction_length available > prediction_length      pred_len
    ARIMA    s_exception == True          > sp                     24
    ARIMA    s_exception == True          > max_p                  2
    ARIMA    s_exception == True          > max_q                  2

    Notes
    -----
    Model-name normalization is performed as follows:

        1. Convert the model name to lowercase.
        2. Remove underscores (``_``).
        3. Remove hyphens (``-``).

    For example, ``"sf-arima"``, ``"sf_arima"``, and ``"sfarima"`` are all
    normalized to ``"sfarima"`` before model selection.
    """

    from imputegap.recovery.manager import TimeSeries

    model_low = model.lower()
    mdl = model_low.replace('_', '').replace('-', '')

    if pred_len is not None:
        if "pred_len" in params:
            params["pred_len"] = pred_len
    if pred_len is not None:
        if "config" in params and "prediction_length" in params["config"]:
            params["config"]["prediction_length"] = pred_len
            print(f"\tprediction_length adapted with the horizon: {params["config"]["prediction_length"]}\n")
    if s_exception and mdl == "arima":
        params["sp"] = 24
        params["max_p"] = 2
        params["max_q"] = 2
        #print(f"\t\t\tARIMA patch up: {params=}")

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
        import warnings
        warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMRegressor was fitted with feature names")
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
    elif mdl == "ltsf":
        from sktime.forecasting.ltsf import LTSFLinearForecaster
        forecaster = LTSFLinearForecaster(**params)
    elif mdl == "naive":
        from sktime.forecasting.naive import NaiveForecaster
        forecaster = NaiveForecaster(**params)

    # new ones -
    elif mdl == "dlinear":
        from darts.models import DLinearModel
        forecaster = DLinearModel(**params)
    elif mdl == "nlinear":
        from darts.models import NLinearModel
        forecaster = NLinearModel(**params)
    elif mdl == "chronos":
        from darts.models import Chronos2Model
        forecaster = Chronos2Model(**params)
    elif mdl == "patchtst":
        from sktime.forecasting.patch_tst import PatchTSTForecaster
        forecaster = PatchTSTForecaster(**params)
    elif mdl == "moment":
        from sktime.forecasting.hf_momentfm_forecaster import MomentFMForecaster
        forecaster = MomentFMForecaster(**params)

    else:
        raise ValueError(f"\n(DOWN) Forecasting model '{model}' not recognized, please choose your algorithm on this list :\n\t{TimeSeries().forecasters}\n")
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
    See reconstruction_window_based() to restore the imputed matrix to its original shape.

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


def get_dataset_imputations(directory="datasets/", fallback_tried=False):
    """
    List the names of all TXT datasets in a directory, excluding subdirectories and files containing ``test``, ``x_``, or ``y_``.

    Parameters
    ----------
    directory : str, optional
        Directory to scan for dataset files.
        Default is "datasets/".

    fallback_tried : bool, optional
        Whether the fallback directory has already been tried.
        Default is False.

    Returns
    -------
    list[str]
        Sorted list of dataset names without the '.txt' extension.

    Example
    -------
    $ names = get_dataset_names("datasets/")

    """
    from pathlib import Path

    print(f"datasets called from: {directory}")

    d = Path(directory)
    bases = set()

    for p in d.glob("*.txt"):
        if not p.is_file():
            continue

        name = p.stem

        if any(excluded in name.lower() for excluded in ("test", "x_", "y_")):
            continue

        bases.add(name)

    if not bases and not fallback_tried:
        return get_dataset_imputations("./imputegap/datasets/", fallback_tried=True)

    return sorted(bases)

def get_dataset_forecasters(directory="datasets/forecast/", fallback_tried=False):
    """
    List the names of the files (txt) inside a directory

    Parameters
    ----------
    directory : str, optional
        Relative path (from the inferred project root) to scan
        Default is "datasets/forecast/".

    Returns
    -------
    list[str]
        Sorted list the names of the files (txt) inside a directory

    Example
    -------
        $ names = get_txt_files("datasets/forecast/")
    """
    from pathlib import Path

    print(f"datasets called from: {directory}")

    d = Path(directory)
    bases = set()

    for p in d.glob("*.txt"):
        if not p.is_file():
            continue

        name = p.name  # e.g., "airq_matrix.txt"
        if name.endswith("_matrix.txt"):
            name = name[:-len("_matrix.txt")]
        elif name.endswith("_season.txt"):
            name = name[:-len("_season.txt")]

        bases.add(name)

    if not bases and not fallback_tried:
        return get_dataset_forecasters("./imputegap/datasets/forecast/", fallback_tried=True)

    return sorted(bases)

def get_datasets_classifiers(directory="datasets/classify/", verbose=True):
    """
    List immediate subdirectory names within a target directory relative to the project root.

    Parameters
    ----------
    directory : str, optional
        Relative path (from the inferred project root) to scan for subdirectories.
        Default is "datasets/classify/".

    verbose : bool, optional
        If True, prints the resolved path and the discovered directory names. Default is True.

    Returns
    -------
    list[str]
        Sorted list of immediate subdirectory names found in the target path.

    Example
    -------
        $ names = get_directory_names("datasets/classify/", verbose=False)
    """
    from pathlib import Path
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = Path(os.path.join(here, directory))

    names = sorted([child.name for child in path.iterdir() if child.is_dir()])

    if verbose:
        print(f"{names =}\n")

    return names

def prepare_series_by_class(labels, raw_data, imputer="cdrec", pattern="mcar", ts_m=None, params=None, rate_dataset=0.4, rate_series=0.4, offset=0.05, imputation_by_class=True, task="default", verbose=True, deep_verbose=False):
    """
    Impute/recover time series data class-by-class, then rebuild the full matrix in the original series order.

    This utility groups series (columns) by their class label, applies a contamination/missingness pattern per class
    (or uses a precomputed missingness mask), imputes each class sub-matrix independently using the chosen imputer,
    and finally stitches all class reconstructions back into a full reconstruction matrix that matches the original
    `raw_data` column ordering.

    Parameters
    ----------
    labels : array-like
        Class labels for each series/instance. Must align with the series axis in `raw_data` (i.e., one label per column).
        Labels are cast to `str` internally to ensure consistent dictionary keys.

    raw_data : np.ndarray
        Original data matrix of shape (N, M) where:

        - N is the number of timestamps/values per series (rows),
        - M is the number of series/instances (columns).

    imputer : str, optional
        Name of the imputation/recovery algorithm to use (passed to `config_impute_algorithm`). Default is "cdrec".

    pattern : str, optional
        Missingness pattern used when `ts_m` is not provided. Passed to `config_contamination`, except for the special
        case "aligned_one" which uses `GenGap.aligned(...)`. Default is "mcar".

    ts_m : np.ndarray or None, optional
        Optional precomputed missingness/contaminated matrix with the same shape as ``raw_data`` (N, M).
        If provided, per-class missingness is taken from `ts_m[:, idxs]` and `pattern/rate_*` are ignored for mask creation.
        Default is None.

    params : dict or None, optional
        Optional imputer hyperparameters passed to `alg.impute(params=params)`. If None, calls ``alg.impute()`` with defaults.
        Default is None.

    rate_dataset : float, optional
        Dataset-level missingness rate used by `config_contamination` when `ts_m` is None. Default is 0.4.

    rate_series : float, optional
        Series-level missingness rate used by `config_contamination` (and by the "aligned_one" case) when `ts_m` is None.
        Default is 0.4.

    offset : float, optional
        Offset/horizon parameter forwarded to aligned contamination generation. Default is 0.1.

    imputation_by_class : bool, optional
        Whether to perform imputation independently for each class or on the complete dataset (default is True). Contamination is applied by class regardless of this setting.

    task: string, optional
            Task of the model (default is "default"). It will load the default parameters for the specific task. "default" for regular imputation / "classification" for classification. / "forecasting" for forecasting.

    verbose : bool, optional
        If True, prints class breakdown, shapes, and progress logs. Default is True.

    Returns
    -------
    tuple
        (alg, full_reconstruction, full_miss)

        where:

        - alg : object
            The last configured imputer instance (from the last processed class), updated so that:

            - ``alg.incomp_data`` = full_miss
            - ``alg.recov_data``  = full_reconstruction
            - ``alg.verbose``     = verbose

        - full_reconstruction : np.ndarray
            Reconstructed/imputed full matrix of shape (N, M), matching the original `raw_data` column ordering.

        - full_miss : np.ndarray
            Full missing/contaminated matrix of shape (N, M) corresponding to the masks used per class.

    Example
    -------
    .. code-block:: python

        alg, X_rec, X_miss = prepare_series_by_class(
                 labels=y_train,
                 raw_data=X_train,
                 imputer="cdrec",
                 pattern="mcar",
                 rate_series=0.2,
                 verbose=False,
        )
    """
    class_to_indices = {}

    #### class the series
    for i, label in enumerate(labels):
        # if y_train is a numpy array, label might be a numpy scalar; str(...) makes it a normal string
        label = str(label)
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(i)

    #### class the series
    class_matrices = {}
    class_miss = {}
    for label, idxs in class_to_indices.items():
        class_matrices[label] = np.array(raw_data[:, idxs])  # rows = idxs, all columns
        if ts_m is not None:
            class_miss[label] = np.array(ts_m[:, idxs])  # rows = idxs, all columns

    class_recon = {}
    class_missing = {}

    # ====================================================================
    # pipiline for imputation by class (with contamination by class)
    # ====================================================================

    # handling each class separately
    # __________________________________________________
    for class_label, X in class_matrices.items():
        alg = None

        # contaminate every class equally
        # __________________________________________________
        if ts_m is None:
            if verbose:
                print(f"\n\t\t\tcontamination by class ({pattern=})")
            missing = config_contamination(ts=X, pattern=pattern, series_rate=rate_series, dataset_rate=rate_dataset, offset=offset, verbose=verbose)

        # else-wise registering the class with its own contamination
        # __________________________________________________
        else:
            if verbose:
                print(f"\n\t\t\tcontamination by dataset ({pattern=})")
            missing = class_miss[class_label]

        if verbose:
            print(f"\nlabel {class_label} - number of series {missing.shape[1]}\n\tdata shape : {X.shape}\n\tmissing matrix shape = {missing.shape}\n")

        # import imputation configuration and load it
        # __________________________________________________
        alg = config_impute_algorithm(incomp_data=missing, algorithm=imputer, task=task, verbose=verbose)
        alg.verbose = verbose
        alg.task = task

        # check and management of the imputation
        # __________________________________________________
        if np.any(np.isnan(missing)):

            # check some special conditions for some algorithms
            # __________________________________________________
            if (rate_series > 0.6 or rate_dataset > 0.6) and imputer in ["TimesNet", "GPT4TS", "NuwaTS"]:
                if verbose:
                    print(f"\t(INFO): High contamination detected for DeepLearning/LLMs model ({rate_dataset=}/{rate_series=}), training rate increased.")
                if params is not None:
                    if imputation_by_class:
                        alg.impute(tr_ratio=0.90, params=params)
                else:
                    if imputation_by_class:
                        alg.impute(tr_ratio=0.90)

            # regular algorithms
            # __________________________________________________
            else:

                # defined params
                # __________________________________________________
                if params is not None:
                    if imputation_by_class:
                        alg.impute(params=params)

                # default params
                # __________________________________________________
                else:
                    if imputation_by_class:
                        alg.impute()

            # manual check and visualisation
            # __________________________________________________
            if deep_verbose and False:
                print(f"\nts.data")
                for i in X:
                    for values in i:
                        print(f"{round(values, 1)}", end="\t\t", sep="\t\t")
                    print("")
                print(f"\nmissing")
                for i in missing:
                    for values in i:
                        if np.isnan(values):
                            print(f"NaN", end="\t\t", sep="\t\t")
                        else:
                            print(f"{round(values, 1)}", end="\t\t", sep="\t\t")
                    print("")
                print(f"\nimputer.recov_data")
                for i in alg.recov_data:
                    for values in i:
                        print(f"{round(values, 1)}", end="\t\t", sep="\t\t")
                    print("")

            if deep_verbose or verbose:
                alg.score(X, alg.recov_data)
                from imputegap.recovery.manager import TimeSeries
                ts = TimeSeries()
                print("\n")
                ts.print_results(alg.metrics)

        # if no imputation asked or needed
        # __________________________________________________
        else:
            if verbose:
                print(f"\tNo missing values detected in the class {class_label}, skip imputation.\n")
            alg.recov_data = X

        # save reconstruction matrix and informative contamination matrix for each class
        # _______________________________________________________________________________
        class_recon[class_label] = alg.recov_data  # shape (len(idxs), 470)
        class_missing[class_label] = missing  # shape (len(idxs), 470)

    # reconstruct the full matrix in original contaminated matrix order
    # _______________________________________________________________________________
    full_reconstruction = np.empty_like(raw_data, dtype=float)
    full_miss = np.empty_like(raw_data, dtype=float)

    for class_label, idxs in class_to_indices.items():
        full_reconstruction[:, idxs] = class_recon[class_label]
        full_miss[:, idxs] = class_missing[class_label]

    if verbose:
        print(f"{full_reconstruction.shape = }")  # (30, 470)
        print(f"{full_miss.shape = }")  # (30, 470)

    if imputation_by_class:
        alg.incomp_data = full_miss
        alg.recov_data = full_reconstruction
        alg.verbose = verbose

    # ====================================================================
    # pipeline for imputation by dataset (with contamination by class)
    # ====================================================================
    else:
        print(f"IMPUTATION BY DATASET")
        alg.incomp_data = full_miss
        alg.impute(params=params)
        full_reconstruction=alg.recov_data
        alg.score(raw_data, alg.recov_data)
        alg.verbose = verbose


    return alg, full_reconstruction, full_miss




def prepare_caching(dataset, algorithm, pattern, x, contamination_by_class, imputation_by_class, fixed_rate, dir="_caching"):
    """
    Prepare the cache identifier and cache directory for an experiment.

    The cache name is constructed from the dataset, contamination pattern,
    imputation algorithm, contamination rates, and task configuration. Algorithm
    names are also normalized and, when applicable, versioned or marked as
    optimized to distinguish different experimental configurations.

    Parameters
    ----------
    dataset : str
        Name of the dataset. Hyphens are removed and the name is converted to
        lowercase when constructing the cache identifier.

    algorithm : str
        Name of the imputation algorithm. Some algorithms are internally renamed
        or versioned for caching purposes, for example ``"moment"`` to
        ``"moment3"`` and ``"missnet"`` to ``"missnet2"``. Algorithms using
        optimized parameters may additionally receive the ``"_optimized"``
        suffix depending on the task. This configuration is made to match the result
        of the CleanImp paper and will only affect the name of the file.

    pattern : str
        Name of the contamination pattern used in the experiment.

    x : float
        Variable contamination rate. The value is converted to an integer
        percentage when constructing the cache name. For example, ``0.4``
        becomes ``"40"``.

    contamination_by_class : bool or None
        Defines the experimental task and contamination configuration.
        - ``True``: classification task with contamination applied by class.
        - ``False``: classification task with contamination applied by dataset.
        - ``None``: forecasting task.

    imputation_by_class : bool or int
        Defines how imputation is performed.
        For classification tasks:
        - ``True``: imputation is performed independently by class.
        - ``False``: imputation is performed on the complete dataset.
        For forecasting tasks, the value represents the forecasting horizon and
        is included in the cache name as ``"horizon<value>"``.

    fixed_rate : float
        Fixed contamination rate used in the experiment. The value is converted
        to an integer percentage when constructing the cache name.

    dir : str, default="_caching"
        Name or relative path of the cache directory. The directory is created
        automatically if it does not already exist.

    Returns
    -------
    name : str
        Unique cache identifier describing the experimental configuration. The
        identifier has the general form::

            <dataset>_<pattern>_<algorithm>_<rate>_<fixed_rate>_<contamination>_<imputation>

    dir_cache : str
        Absolute path to the cache directory.

    Notes
    -----
    Algorithm names may be modified internally to distinguish specific
    implementations, versions, or optimized configurations. The optimization
    suffix is applied differently for classification and forecasting tasks.
    """
    classification_task = True
    optizer = True
    if contamination_by_class is not None:
        if contamination_by_class:
            c = "contbyclass"
        else:
            c = "contbydataset"
        if imputation_by_class:
            i = "imputationbyclass"
        else:
            i = "imputationbydataset"

    elif contamination_by_class is None and imputation_by_class is None:
        c = "upstream"
        i = ""

    else:
        c = "forecasting"
        i = "horizon"+str(imputation_by_class)
        classification_task = False


    if algorithm.lower() == "moment":
        algorithm = "moment3"
    if algorithm.lower() == "missnet" or algorithm.lower() == "miss_net" or algorithm.lower() == "miss-net":
        algorithm = "missnet2" #m_versionning
    if algorithm.lower() == "iim":
        algorithm = "iim"
    if algorithm.lower() == "stmvl":
        algorithm = "stmvl2"
    if algorithm.lower() == "xgboost":
        algorithm = "xgboost"
    if algorithm.lower() == "SoftImpute":
        algorithm = "softimpute3"

    if False:#if optizer:
        s_add = "_optimized"
        if classification_task:
            if algorithm.lower() == "iterativesvd" or algorithm.lower() == "iterative_svd" or algorithm.lower() == "iterative-svd":
                algorithm="iterativesvd"+s_add
            if algorithm.lower() == "cdrec":
                algorithm="cdrec"+s_add
            if algorithm.lower() == "grouse":
                algorithm="grouse"+s_add
            #if algorithm.lower() == "svt":
            #    algorithm="svt"+s_add
            if algorithm.lower() == "iim":
                algorithm="iim"+s_add
            if algorithm.lower() == "csdi":
                algorithm="csdi"+s_add
            if algorithm.lower() == "pristi":
                algorithm="pristi"+s_add
        else:
            if algorithm.lower() == "iterativesvd" or algorithm.lower() == "iterative_svd" or algorithm.lower() == "iterative-svd":
                algorithm="iterativesvd"+s_add
            if algorithm.lower() == "grouse":
                algorithm="grouse"+s_add
            if algorithm.lower() == "svt":
                algorithm="svt"+s_add
            if algorithm.lower() == "dynammo":
                algorithm="dynammo"+s_add
            if algorithm.lower() == "xgboost":
                algorithm="xgboost"+s_add
            if algorithm.lower() == "saits":
                algorithm="saits"+s_add
            if algorithm.lower() == "csdi":
                algorithm="csdi"+s_add
            if algorithm.lower() == "pristi":
                algorithm="pristi"+s_add

    dataset_s = dataset.replace("-", "")
    algorithm_s = algorithm.replace("-", "")
    cont_rate = str(int(x * 100))
    fixed_rate = str(int(fixed_rate * 100))

    name = dataset_s.lower() + "_" + pattern.lower() + "_" + algorithm_s.lower() + "_" + cont_rate + "_" + fixed_rate + "_" + c + "_" + i
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dir_cache = os.path.join(here, dir)
    os.makedirs(dir_cache, exist_ok=True)

    return name, dir_cache


def ts_caching_save(data, type="classification", artifact=None, name="", dir_cache="", verbose=True):
    """
    Save time-series data to a cache file using a task-specific format.

    For classification tasks, the data is saved in the UEA/UCR ``.ts`` format,
    including the corresponding class labels. For other task types, the data is
    saved as a plain-text matrix with space-separated values.

    Parameters
    ----------
    data : array-like
        Time-series data to save.

        For classification, ``data`` is expected to be a 2D matrix where rows
        represent time points and columns represent individual time series. The
        matrix is transposed before being written so that each output row
        represents one complete time series.

        For other task types, ``data`` is written directly without
        transposition, with one row per line.

    type : str, default="classification"
        Type of task or cached artifact.

        - ``classification`` saves the data in UEA/UCR ``.ts`` format using
          the filename suffix ``"_TRAIN.ts"``.
        - Any other value saves the data as a space-separated text file using
          the suffix ``"_<type>.txt"``.

    artifact : array-like, optional
        Additional information associated with the cached data. For
        classification tasks, this must contain the class label corresponding
        to each time series. It is not used for other task types.

    name : str, default=""
        Base name of the cache file. For classification, the resulting filename
        is ``<name>_TRAIN.ts``. Otherwise, it is ``<name>_<type>.txt``.

    dir_cache : str, default=""
        Directory in which the cache file is saved.

    verbose : bool, default=True
        Whether to enable verbose output. Currently unused by this function.

    Returns
    -------
    str
        Path to the generated cache file.

    Notes
    -----
    Classification files are written as univariate, equal-length time series without timestamps or missing values. The class labels declared in the ``.ts`` header preserve their order of first occurrence in ``artifact``. For non-classification files, values that cannot be converted to ``float`` or that evaluate to ``NaN`` are written as ``NaN``.
    """

    if type == "classification":
        extention = "_TRcAIN.ts"
        out_dir = os.path.join(dir_cache, name+extention)

        y = artifact
        X = data.T
        uniq = []
        seen = set()
        for lab in y.tolist():
            if lab not in seen:
                seen.add(lab)
                uniq.append(lab)

        # Infer univariate/equalLength from your described format (2D matrix)
        header = [
            f'#"IMPUTEGAP: CACHING..."',
            f"@problemName {name}",
            "@timeStamps false",
            "@missing false",
            "@univariate true",
            "@equalLength true",
            f"@seriesLength {X.shape[1]}",
            "@classLabel true " + " ".join(l for l in uniq),
            "@data",
        ]

        with open(out_dir, "w", encoding="utf-8", newline="\n") as f:
            for ln in header:
                f.write(ln + "\n")
            for i in range(X.shape[0]):
                row = X[i]
                series_part = ",".join(str(float(v)) for v in row)
                f.write(f"{series_part}:{y[i]}\n")

    else:
        import math
        extention = "_"+type+".txt"
        out_dir = os.path.join(dir_cache, name + extention)
        X = data
        def fmt(v):
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return "NaN"
            return "NaN" if math.isnan(fv) else str(fv)

        with open(out_dir, "w", encoding="utf-8", newline="\n") as f:
            for i in range(X.shape[0]):
                row = X[i]
                line = " ".join(fmt(v) for v in row)
                f.write(line + "\n")

    return out_dir


def classifiers_caching(y_pred, model="", t="", name="", dir_cache="", family="_c", ext="_PRED.txt", x=0):
    """
    Save classifier or forecaster predictions to a cache file.

    The output subdirectory is selected from ``family``. Prediction values are
    written one per line, and some model names are internally versioned before
    constructing the cache filename.

    Parameters
    ----------
    y_pred : array-like
        Predicted values or class labels to save. The input is converted to a
        NumPy array before being written to disk.

    model : str, default=""
        Name of the classification or forecasting model. Some model names are
        internally renamed for cache versioning, including:
        - ``"cif"`` to ``"cif2"``
        - ``"chronos"`` to ``"chronos7d"``
        - ``"patchtst"`` to ``"patchtst4"``
        this changes are made to match the version of algorithm from the CleanImp paper

    t : str, default=""
        Additional task or experiment identifier included in the cache
        filename.

    name : str, default=""
        Base experiment or dataset cache name included in the output filename.

    dir_cache : str, default=""
        Root cache directory. A ``classifiers/`` or ``forecasters/``
        subdirectory is created automatically inside this directory.

    family : str, default="_c"
        Model family identifier used both in the filename and to determine the
        output subdirectory.

        If ``family`` contains ``"c"``, predictions are stored under
        ``classifiers/``. Otherwise, they are stored under ``forecasters/``.

    ext : str, default="_PRED.txt"
        File extension or suffix appended to the generated cache filename.

    x : int or float, default=0
        Additional experiment parameter. Currently unused by this function.

    Returns
    -------
    out_path : str
        Full path to the generated prediction cache file.

    s_name : str
        Generated cache filename, following the format::

            <family>_<model>_<t>_<name><ext>

    Notes
    -----
    Each prediction is written on a separate line.

    The destination directory is created automatically if it does not already
    exist.
    """
    if "c" in family:
        d = "classifiers/"
    else:
        d = "forecasters/"

    if model.lower() == "cif":
        model = "cif2"
    elif model.lower() == "chronos":
        model = "chronos7d"
    elif model.lower() == "patchtst":
        model = "patchtst4"

    dir_cache = os.path.join(dir_cache, d)
    os.makedirs(dir_cache, exist_ok=True)
    s_name = f"{family}_{model}_{t}_{name}{ext}"
    out_path = os.path.join(dir_cache, s_name)
    y_pred = np.asarray(y_pred)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for p in y_pred:
            f.write(f"{p}\n")
    return out_path, s_name

def pred_caching_load(path):
    """
    Load cached predictions from a text file.

    The cache file is expected to contain one prediction per line. Empty lines
    are ignored, and all loaded values are returned as strings in a one-
    dimensional NumPy array with ``dtype=object``.

    Parameters
    ----------
    path : str or path-like
        Path to the cached prediction file.

    Returns
    -------
    numpy.ndarray
        One-dimensional NumPy array containing the cached predictions.
        Values are preserved as strings and the returned array uses
        ``dtype=object``.

    Notes
    -----
    No automatic type inference or numeric conversion is performed. Predictions
    such as integers, floating-point values, or class labels are all loaded as
    strings.

    Trailing newline characters are removed, and empty lines are skipped.
    """
    with open(path, "r", encoding="utf-8") as f:
        vals = [ln.rstrip("\n") for ln in f if ln.strip() != ""]
    return np.array(vals, dtype=object)

# def generate_report_downstream(algorithms=["MeanImpute"], bypass_error=True, division_dataset=3, inc=0, verbose=True):
#     from imputegap.recovery.benchmark import Benchmark
#     parts = split_performance_classifier(names=get_datasets_classifiers(), n_parts=division_dataset)
#     for a in algorithms:
#         for p in parts:
#             inc = inc + 1
#             bench = Benchmark()
#             bench.eval_downstream_classification(classifiers=["arsenal"],
#                                                  algorithms=[a],
#                                                  datasets=p,
#                                                  patterns=["mcar", "aligned_series", "aligned_timestamps"],
#                                                  x_axis=[0.1, 0.2, 0.4, 0.6, 0.8],
#                                                  metrics=["RMSE", "DOWNSTREAM_ACC", "MI", "CORRELATION", "RUNTIME"],
#                                                  normalizer="z-score",
#                                                  report_title=str(inc) + "_REPORTS_score_" + str(a) + "_",
#                                                  contamination_by_class=True,
#                                                  imputation_by_class=True,
#                                                  bypass_error=bypass_error,
#                                                  to_cache=True,
#                                                  use_cache=True,
#                                                  run_downstream=False,
#                                                  fixed_rate=0.2,
#                                                  inner_plots=False,
#                                                  verbose=False,
#                                                  plots=False)
#         if verbose:
#             print("end")
#
#
# def generate_report_forecasting(algorithms=None, bypass_error=True, division_dataset=3, inc=0, verbose=True):
#     from imputegap.recovery.benchmark import Benchmark
#     parts = split_performance_classifier(names=get_dataset_forecasters(directory='datasets/forecast/'), n_parts=division_dataset)
#     if algorithms is None:
#         algorithms = list_of_algorithms()
#     for a in algorithms:
#         for p in parts:
#             inc = inc + 1
#             bench = Benchmark()
#             bench.eval_downstream_forecasting(forecasters=["exp-smoothing"],
#                                               horizons=[12],
#                                               algorithms=[a],
#                                               datasets=p,
#                                               patterns=["mcar", "aligned_series", "aligned_timestamps"],
#                                               x_axis=[0.1, 0.2, 0.4, 0.6, 0.8],
#                                               metrics=["RMSE", "DOWNSTREAM_SMAPE", "MI", "CORRELATION", "RUNTIME"],
#                                               normalizer="z-score",
#                                               report_title=str(inc) + "_REPORTS_score_" + str(a) + "_",
#                                               bypass_error=bypass_error,
#                                               to_cache=True,
#                                               use_cache=True,
#                                               run_downstream=False,
#                                               evaluate_upstream=True,
#                                               fixed_rate=0.2,
#                                               inner_plots=False,
#                                               verbose=False,
#                                               plots=False,
#                                               nbr_series=10000,
#                                               nbr_vals=10000)
#         if verbose:
#             print("end")


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


def auto_seq_llms(data_x, goal="seq", subset=False, high_limit=200, low_limit=0, exception=False, b=None, verbose=True, deep_verbose=False):
    """
    Heuristic brute-force search for a "good" (seq_len, batch_size) pair for sliding-window training.

    The function searches over candidate sequence lengths (`seq_len`) and batch sizes (`batch_size`) and
    scores each pair using a simple heuristic that tries to minimize leftover windows ("remainders") when
    batching sliding windows. It optionally enforces that the chosen batch size is feasible for multiple
    subsets (e.g., train/test/val-style splits).

    Parameters
    ----------
    data_x : np.ndarray
        Input array. Expected shape is (T, F, ...) or (T, F). The first dimension `T` is treated as the
        time axis (number of timestamps). The second dimension `F` is treated as the feature/series axis.

    goal : str, optional
        Optimization objective controlling the scoring heuristic:

        - "seq"      : prioritize larger seq_len (after feasibility), then balance with batch_size
        - "batch"    : prioritize larger batch_size (after feasibility), then balance with seq_len
        - "low_limit": special mode used when `low_limit != 0`, favoring smaller seq_len near constraints
        - anything else falls back to a "balance" style score

        Default is "seq".

    subset : bool, optional
        If True, the function evaluates feasibility on multiple subset sizes (currently full length T and
        a "train" length Tr = 0.7*T). The idea is to ensure that a (seq_len, batch_size) choice yields at
        least 2 sliding windows in each subset, and to consider remainders across subsets.
        If False, feasibility is checked on [T, T//3]. Default is False.

    high_limit : int, optional
        Upper bound for the starting `seq_len` search (acts like a cap). If `high_limit` is larger than
        10% of T and `exception` is False, it is reduced to int(0.1*T). Default is 200.

    low_limit : int, optional
        Lower bound used to restrict the minimum seq_len considered. If non-zero, the function switches into
        a special mode:

        - starting_point becomes T
        - start_batch becomes 8
        - max_batch becomes 32
        - goal becomes "low_limit"

        Default is 0.

    exception : bool, optional
        If True, prevents auto-capping `high_limit` to 10% of T. Default is False.

    b : any, optional
        Optional flag that changes how max_batch and start_batch are set:

        - if b is not None, max_batch becomes F//2 (where F is data_x.shape[1])
        - and start_batch is forced to 2

        Default is None.

    verbose : bool, optional
        If True, prints the selected best pair and some diagnostics. Default is True.

    deep_verbose : bool, optional
        If True, prints per-candidate diagnostic lines during search. Default is False.

    Returns
    -------
    tuple[int | None, int | None]
        (seq_len, batch_size) for the best-scoring candidate. Returns (None, None) if no valid candidates exist.

    Example
    -------
        >>> seq_len, batch_size = auto_seq_llms(data_x, goal="seq", subset=True, high_limit=128)
        >>> print(seq_len, batch_size)
    """

    if goal == "llm":
        if data_x.shape[0] > 48:
            return 48, 48
        else:
            s = 1 << (high_limit.bit_length() - 1)  # largest power of 2 <= size
            return s, s
    if goal == "moment":
        s = np.array([1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024])
        idx = np.searchsorted(s, high_limit, side="left")
        s = int(s[min(idx, len(s) - 1)])
        if data_x.shape[0] < s:
            s = 1 << (high_limit.bit_length() - 1)  # largest power of 2 <= size
        return s, s

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

    if low_limit != 0:
        starting_point = data_x.shape[0]
        start_batch = 8
        max_batch = 32
        goal = "low_limit"

    min_seq_len = max(2, int(low_limit) + 1) if low_limit else 2
    candidates = []

    for seq_len in range(starting_point, min_seq_len-1, -1):
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
            elif goal == "low_limit":
                # prefer perfect match first, then larger batch_size
                score = 1000 * (total_remainder > 0) + (2*seq_len) + abs(seq_len - batch_size)
            else:  # "balance"
                # keep original spirit, but on aggregated remainder
                score = total_remainder + abs(seq_len - batch_size) * 0.1

            # store a copy of list_windows for this candidate
            candidates.append((score, seq_len, batch_size, list_windows.copy()))

    if not candidates:
        seq_len = low_limit + 1
        batch_size = 16
        return seq_len, batch_size

    # pick the combination with the lowest score
    candidates.sort(key=lambda x: x[0])
    best = candidates[0]
    score, seq_len, batch_size, best_windows = best

    return seq_len, batch_size


def max_consecutive_nans_per_col_loop(a):
    a = np.asarray(a)
    nan = np.isnan(a)
    out = np.zeros(nan.shape[1], dtype=int)

    for j in range(nan.shape[1]):
        x = nan[:, j]
        # run-length via diff
        d = np.diff(np.r_[False, x, False].astype(np.int8))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        out[j] = (ends - starts).max(initial=0)

    return out


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
    incomp_data : np.ndarray
        Input data matrix containing potentially missing values (NaNs).

    Returns
    -------
    float
        Proportion of missing values in the input matrix, ranging from 0.0 to 1.0.
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
        Lower bound of the acceptable percentage range (default is 0.001).

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
    - If the percentage is outside the accepted range and cannot be converted from a percentage, a ValueError is raised.
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

        Parameters
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
            Whether to replace masked values with the specified sentinel value
            to prevent information leakage during model training.

        offset : float, default=0.05
            Minimum temporal offset in the begining of the series

        block_selection : bool, default=True
            Whether to simulate missing values in contiguous blocks (True) or randomly (False).

        seed : int, default=42
            Seed for NumPy random number generation to ensure reproducibility.

        verbose : bool, default=False
            Whether to print logging/debug information during execution.

        Returns
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



def control_boundaries(rank=2, boundary=3, algorithm="Algorithm", reduction=1):
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
        print(f"\t\t\t\t(WARNING) {algorithm} ! rank {rank} is higher than the number of series {boundary}. Reduce to {new_estimator}.")

        if new_estimator > 0:
            return new_estimator
        else:
            print(f"\t\t\t(ERROR) {algorithm}\n\tNot enough series to impute with this algorithm {boundary} <= 0.\n")
            return 1
    else:
        return rank



def list_of_upstream_classifiers_algorithms():
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
    return [
        "MeanImpute",
        "TRMF",
        "Dynammo",
        "MICE",
        "MissNet",
        "GPT4TS",
        "Interpolation"]


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
        "SPIRIT", # temp_to_put_back
        "SoftImpute",
        "SVT",
        "TRMF",
        "STMVL",
        "DynaMMo",
        "TKCM", # temp_to_put_back
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
        "CSDI",
        "Moment"
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


def list_of_top_cleanimp_cl_algorithms():
    return sorted([
        "Moment",
        "GRIN",
        "MICE",
        "Dynammo",
        "TRMF",
        "MeanImpute"
    ])
def list_of_top_cleanimp_for_algorithms():
    return sorted([
        "GPT4TS",
        "SAITS",
        "MICE",
        "Dynammo",
        "SoftImpute",
        "MeanImpute"
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

def list_of_classifiers():
    """
    Return the list of available downstream models for classifiers from ImputeGAP.

    Parameters
    ----------
    None

    Returns
    -------
    list of str
       A sorted list of downstream models names supported by ImputeGAP.
    """
    return sorted([
        #"forest",
        #"muse",
        "weasel",
        "itde",
        #"tde",
        "cboss",
        "knn",
        #"proxforest",
        #"proxtree",
        "proxstump",
        "tsf",
        "cif",
        "stc",
        "lstm",
        "cnn",
        "svc",
        "arsenal",
        #"rocket",
        "catch22",
        #"mpc",
        "signature",
        "tsfresh",
        "shapedtw",
    ])

def list_of_forecasters():
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
        "ltsf",
        #"sf-arima",
        ###"bats", > not working
        #"ets",
        "croston",
        #"theta",
        #"unobs",
        #"naive"
        "patchtst",
        "moment"
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
        "transformer",
        "dlinear",
        "nlinear",
        "chronos",
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
        "LLMs.GPT4TS",
        "LLMs.Moment"
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


def dataset_size_features(X, y):
    """
    Dataset characterization (shape + label statistics).

    Parameters
    ----------
    X : np.ndarray
        Time-series array. Accepted shapes:
        - (n_series, n_timestamps) or (n_timestamps, n_series)
    y : np.ndarray
        Class labels of shape (n_series,).

    Returns
    -------
    tuple
        (n_series, n_timestamps, n_classes, mean_per_class, min_per_class, max_per_class)
    """
    n_timestamps, n_series = int(X.shape[0]), int(X.shape[1])

    classes, counts = np.unique(y, return_counts=True)
    n_classes = int(classes.size)

    mean_per_class = float(np.mean(counts)) if counts.size else float("nan")
    min_per_class = int(np.min(counts)) if counts.size else 0
    max_per_class = int(np.max(counts)) if counts.size else 0

    return n_series, n_timestamps, n_classes, mean_per_class, min_per_class, max_per_class

def corr_dataset_features(X, y):
    """
    Compute correlation-based dataset features and a simple silhouette-like index.

    It also computes:
      - silhouette : float
        Defined here as abs(corr_in_mean - corr_out_mean), i.e., separation between
        within-class and between-class correlation means.

    Parameters
    ----------
    X : np.ndarray
        Series matrix with shape (n_series, n_timestamps) or (n_timestamps, n_series).
        If X is (n_timestamps, n_series) and len(y) == n_series, X is transposed.
    y : np.ndarray
        Class labels for each series of shape (n_series,).

    Returns
    -------
    tuple
        (silhouette, corr_in_mean, corr_out_mean, features, raw) where:

        - silhouette : float
            abs(corr_in_mean - corr_out_mean)
        - corr_in_mean : float
            Mean correlation among same-class pairs.
        - corr_out_mean : float
            Mean correlation among different-class pairs.
        - features : list[tuple[str, float]]
            Named summary features:
            ("corr-mean", ...), ("corr-stddev", ...),
            ("corr-in-mean", ...), ("corr-in-stddev", ...),
            ("corr-out-mean", ...), ("corr-out-stddev", ...)
        - raw : dict
            Arrays used to compute the statistics:
            similaritiesFlat, similaritiesInClass, similaritiesOutClass, corr_matrix
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    if X.shape[0] != len(y) and X.shape[1] == len(y):
        X = X.T
    elif X.shape[0] != len(y):
        raise ValueError(f"Incompatible shapes: X={X.shape}, y={len(y)}")
    n = X.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 series to form pairs.")

    corr = np.corrcoef(X)
    flat,in_class,out_class = [],[],[]

    for i in range(n):
        for j in range(i + 1, n):
            sim = corr[i, j]
            flat.append(sim)
            if y[i] == y[j]:
                in_class.append(sim)
            else:
                out_class.append(sim)

    flat = np.asarray(flat, dtype=float)
    in_class = np.asarray(in_class, dtype=float) if len(in_class) else np.asarray([], dtype=float)
    out_class = np.asarray(out_class, dtype=float) if len(out_class) else np.asarray([], dtype=float)

    corr_mean, corr_std = (float(np.mean(flat)), float(np.std(flat, ddof=0)))
    corr_in_mean, corr_in_std = (float(np.mean(in_class)), float(np.std(in_class, ddof=0)))
    corr_out_mean, corr_out_std = (float(np.mean(out_class)), float(np.std(out_class, ddof=0)))

    features = [("corr-mean", corr_mean), ("corr-stddev", corr_std), ("corr-in-mean", corr_in_mean),
                ("corr-in-stddev", corr_in_std), ("corr-out-mean", corr_out_mean), ("corr-out-stddev", corr_out_std), ]

    raw = {"similaritiesFlat": flat, "similaritiesInClass": in_class, "similaritiesOutClass": out_class, "corr_matrix": corr, }

    silhouette = abs(corr_in_mean-corr_out_mean)

    return silhouette, corr_in_mean, corr_out_mean, features, raw


def dataset_classifier_features(data, labels, name="", rmse=None, verbose=True):
    """
    Compute a feature row describing a classification dataset.

    This combines:
      - size statistics (n_series, n_timestamps, class counts)
      - correlation-based separation features (inner/out correlation means + silhouette-like index)
      - categorical descriptors (density_of_class, density_of_series, distribution)
      - optional upstream imputation quality (rmse -> upstream_recovery)
      - an English compactness label derived from silhouette_index

    Parameters
    ----------
    data : np.ndarray
        Dataset series array (see `dataset_size_features` for supported shapes).
    labels : np.ndarray
        Class labels of shape (n_series,).
    name : str, optional
        Dataset name to store in the output row (default is "").
    rmse : float, optional
        Optional upstream RMSE to label recovery quality; if None, treated as 0 (default is None).
    verbose : bool, optional
        Whether to print a human-readable summary of computed features (default is True).

    Returns
    -------
    dict
        A dictionary representing one dataset row with fields:
        dataset, n_classes, n_series, n_timestamps, mean/min/max_series_per_class,
        corr_in_mean, corr_out_mean, rmse, density_of_class, density_of_series,
        distribution, upstream_recovery, silhouette_index, compactness.
    """

    n_series, n_timestamps, n_classes, mean_class_per_class, min_class_per_class, max_class_per_class = dataset_size_features(data, labels)
    silhouette, inner, outer, features, raw = corr_dataset_features(data, labels)

    # --- 1) density_of_class (based on n_classes) ---
    if n_classes <= 2:
        density_of_class = "small"
    elif 3 <= n_classes <= 9:
        density_of_class = "medium"
    else:
        density_of_class = "high"
    # --- 2) density_of_series (based on min_series_per_class) ---
    if min_class_per_class <= 10:
        density_of_series = "small"
    elif 11 <= min_class_per_class <= 100:
        density_of_series = "medium"
    else:
        density_of_series = "high"
    # --- 3) distribution ---
    mean_c = float(mean_class_per_class)
    if (min_class_per_class == max_class_per_class) and (abs(mean_c - min_class_per_class) < 1e-12):
        distribution = "distributed"
    else:
        distribution = "similar"

    # --- 4) rmse ---
    if rmse is None:
        rmse = 0
    if rmse == 0:
        upstream_recovery = "-"
    elif rmse <= 0.2:
        upstream_recovery = "good"
    elif 0.2 <= rmse <= 0.7:
        upstream_recovery = "medium"
    elif 0.7 <= rmse <= 1.2:
        upstream_recovery = "bad"
    else:
        upstream_recovery = "terrible"

    # --- 4) compactness label from silhouette_index ---
    if silhouette > 0.25:
        compactness = "compact"
    elif silhouette >= 0.0:
        compactness = "overlapping"
    else:
        compactness = "spread"

    if verbose:
        print(f"\n\n{name}___FEATURE ANALYSIS_____________________________________________")
        print(f"{n_classes=}\n{n_series = }\n{n_timestamps = }\n{mean_class_per_class = }\n{min_class_per_class = }\n{max_class_per_class = }")
        print(f"\nsilhouette index: {silhouette} - {compactness}\n\tinner correlation: {inner}\n\touter correction: {outer}")
        print(f"\n{density_of_class=}\n{density_of_series=}\n{distribution=}\n{rmse=} > {upstream_recovery}")

    row = {"dataset": name, "n_classes": n_classes, "n_series": n_series, "n_timestamps": n_timestamps,
        "mean_series_per_class": round(mean_class_per_class, 1),
        "min_series_per_class": min_class_per_class,
        "max_series_per_class": max_class_per_class,
        "corr_in_mean": round(inner, 6),
        "corr_out_mean": round(outer, 6),
        "rmse": rmse,
        "density_of_class": density_of_class,
        "density_of_series": density_of_series,
        "distribution": distribution,
        "upstream_recovery": upstream_recovery,
        "silhouette_index": round(silhouette, 6),
        "compactness": compactness,
    }

    return row


def save_dataset_classifier_features(datasets, save_dir = "./imputegap_assets/features", upstream=False, algorithm="SoftImpute", pattern="mcar", verbose=True):
    """
        Compute and export classifier dataset features for multiple datasets.

        For each dataset name in `datasets`, this function:
          - loads the classification dataset through `TimeSeries` / `Artifact`
          - computes size + correlation features via `dataset_classifier_features`
          - optionally computes upstream RMSE (imputation recovery) if `upstream=True`
          - exports the aggregated table to an Excel file: `{save_dir}/classifiers_dataset_features.xlsx`

        Parameters
        ----------
        datasets : list[str]
            List of dataset identifiers to load and analyze.
        save_dir : str, optional
            Directory where the Excel file is written (default is "./imputegap_assets/features").
        upstream : bool, optional
            If True, run upstream imputation and compute RMSE for each dataset (default is False).
        algorithm : str, optional
            Imputation algorithm name to use when `upstream=True` (default is "SoftImpute").
        pattern : str, optional
            Missingness pattern used for upstream imputation (default is "mcar").
        verbose : bool, optional
            Whether to print per-dataset summaries and export details (default is True).

        Returns
        -------
        pd.DataFrame
            A dataframe containing one row per dataset, sorted by dataset name.
        """

    from recovery.manager import TimeSeries
    from recovery.downstream import Artifact

    rows, rmse = [], None
    for d in datasets:
        ts = TimeSeries(verbose=False)
        artifact = Artifact(dataset=d, model="arsenal", algorithm=algorithm, pattern=pattern)
        ts.load_classify_dataset(artifact.dataset, normalizer=None, to_numpy=True, verbose=False)
        classify_sets = ts.load_proper_output_classify()
        if upstream:
            imputer, recovery, missing = prepare_series_by_class(labels=classify_sets[1], raw_data=ts.data.copy(), imputer=artifact.algorithm, params=None, pattern=artifact.pattern, rate_dataset=artifact.fixed_x, rate_series=artifact.x, offset=artifact.offset, verbose=False)
            imputer.incomp_data = missing
            imputer.score(ts.data, recovery)
            rmse = imputer.metrics["RMSE"]
        rows.append(dataset_classifier_features(data=ts.data, labels=classify_sets[1], name=artifact.dataset, rmse=rmse, verbose=verbose))
    f = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
    out_path = os.path.join(save_dir, f"classifiers_dataset_features.xlsx")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        if verbose:
            f.to_excel(out_path, index=False)
        print(f"\nfeatures saved to {out_path}")
    except Exception as e:
        print(f"(ERROR): save_dataset_classifier_features : Excel export failed ({e}).")

    return f


from typing import List, Tuple

def split_performance_classifier(names: List[str], n_parts: int = 3) -> Tuple[List[List[str]], List[str]]:
    """

    Parameters
    ----------
    names : list[str]
        List of dataset names to split.

    n_parts : int, optional
        Number of chunks to create (default is 3). Must be >= 1.

    Returns
    -------
    tuple (parts, names) where:
        - parts : list[list[str]]
            A list containing `n_parts` sublists (chunks) of `names`.

    Raises
    ------
    ValueError
        If `n_parts` is less than 1.
    """
    from typing import List, Tuple

    if n_parts < 1:
        raise ValueError("n_parts must be >= 1")
    if not names:
        return [[] for _ in range(n_parts)], []

    n = len(names)

    # Compute split indices that evenly partition [0, n] into n_parts
    cuts = [(i * n) // n_parts for i in range(n_parts + 1)]
    parts = [names[cuts[i]:cuts[i + 1]] for i in range(n_parts)]

    return parts


def compute_variables_saitslike(incomp_data, seq_len, batch_size, sliding_windows, tr_ratio, verbose):
    """
    Compute and adapt sequence-related variables for SAITS-like models.

    This function determines whether the input configuration is multivariate and
    automatically computes the sequence length and batch size when these values are
    not explicitly provided.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        Two-dimensional contaminated time-series matrix used to determine the
        sequence and batch configuration.

    seq_len : int
        Sequence length used by the model.
        If set to ``-1``, the sequence length is determined automatically using
        either ``auto_seq_sample`` or ``auto_seq_llms`` depending on the
        multivariate configuration.

    batch_size : int
        Batch size used by the model.
        If set to ``-1``, the batch size is determined automatically together with
        ``seq_len``.

    sliding_windows : int
        Sliding-window configuration used to determine how the input data is
        processed.
        - ``0``: multivariate mode is enabled.
        - Any other value: multivariate mode is disabled.
        In multivariate mode, when automatic sequence selection is performed,
        ``sliding_windows`` is set to the resulting ``seq_len``.

    tr_ratio : float
        Training ratio passed to ``auto_seq_sample`` when automatic configuration
        is performed in multivariate mode.

    verbose : bool
        Whether to display information during automatic sequence and batch-size
        selection.

    Returns
    -------
    multivariate : bool
        Whether multivariate processing is enabled. ``True`` when
        ``sliding_windows == 0`` at function entry, otherwise ``False``.

    seq_len : int
        Final sequence length after optional automatic configuration.

    batch_size : int
        Final batch size after optional automatic configuration.

    sliding_windows : int
        Final sliding-window configuration.

    error : bool
        Indicates whether the resulting sequence length is incompatible with the
        input data.

        ``True`` when::

            seq_len > len(incomp_data)

        otherwise ``False``.

    Automatic configuration
    -----------------------

    MODE          CONDITION                     > SEQ_LEN / BATCH_SIZE
    ---------------------------------------------------------------------------
    Multivariate  seq_len == -1 or batch == -1  > auto_seq_sample(...)
    Other         seq_len == -1 or batch == -1  > auto_seq_llms(..., high=50)

    Additional rules
    ----------------
    Multivariate:
        After automatic configuration, ``sliding_windows = seq_len``.

    All modes:
        If ``seq_len > len(incomp_data)``, ``error`` is set to ``True`` and an
        error message is displayed.
    """

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

def compute_variables_timesnetlike(ts_m, strat, tr_ratio, size_limitation, model="", security=None, high=64, consecutive=False, verbose=True, algo="as"):
    """
    Compute and adapt sequence length and batch size for TimesNet-like models.

    This function determines an initial sequence length and batch size using
    ``auto_seq_llms`` and then adapts these values according to the amount and
    structure of missing data in ``ts_m``.

    Additional model-specific patches are applied for TimesNet and GPT4TS when
    the default sequence configuration is incompatible with the number of
    missing values or with particular dataset dimensions.

    Parameters
    ----------
    ts_m : numpy.ndarray
        Two-dimensional time-series matrix containing the data to process.
        Missing values must be represented by ``NaN``.

        The function uses::

            ts_m.shape[0]

        as the main dataset length ``M`` and ``ts_m.shape[1]`` as the second
        dataset dimension when applying configuration patches.

    strat : str
        Strategy passed to ``auto_seq_llms`` to determine the initial sequence
        length and batch size.

    tr_ratio : float
        Training ratio used when computing the amount of data available for
        sequence construction.

        The same value is returned unchanged by this function.

    size_limitation : bool
        Flag indicating whether the sequence length has already been limited by
        the available dataset size.

        The flag may be changed to ``True`` when the function needs to restrict
        or specially adapt the sequence length.

    model : str, default=""
        Model name used for informational messages when sequence adaptation is
        required.

    security : optional, default=None
        Controls the safety margin between the number of missing values and the
        selected sequence length.

        The input value itself is not preserved. Internally, the function uses:

        - ``1.15`` when ``M < 500``, ``tr_ratio < 0.6``, and ``security`` is
          not ``None``.
        - ``1.05`` in all other cases.

    high : int, default=64
        Initial maximum sequence length passed as ``high_limit`` to
        ``auto_seq_llms``.

    consecutive : bool, default=False
        Determines how the amount of missing data, ``taux_nans``, is computed.

        - ``False``:
          ``taux_nans`` is the maximum total number of NaNs found in any column.
        - ``True``:
          ``taux_nans`` is the maximum number of consecutive NaNs found in any column.

    verbose : bool, default=True
        Whether to print information about automatic sequence adaptation and
        the final preprocessing configuration.

        Note that some TimesNet/GPT4TS adaptation messages are currently
        printed regardless of this flag.

    algo : str, default="as"
        Algorithm for which the configuration is being generated.

        Additional sequence adaptation is enabled when ``algo`` is:

        - ``"TimesNet"``
        - ``"GPT4TS"``

        GPT4TS also receives additional patches related to its maximum sequence
        length.

    Returns
    -------
    seq_len : int
        Final sequence length after automatic selection and all applicable
        adaptations or model-specific patches.

    batch_size : int
        Final batch size after automatic selection and all applicable
        adaptations.

    size_limitation : bool
        Updated size-limitation flag. This becomes ``True`` when sequence
        selection has to be constrained by the available data or by one of the
        GPT4TS large-missingness patches.

    taux_nans : int
        Maximum amount of missing data detected according to ``consecutive``.

    tr_ratio : float
        Original training ratio, returned unchanged.



    Specific configuration patches
    ------------------------------

    ALGO              SHAPE          CONTAMINATION / NaNs     > SEQ_LEN    BATCH_SIZE
    ---------------------------------------------------------------------------------
    TimesNet / GPT4TS  (176, *)       10                       > 32         16
    TimesNet / GPT4TS  (<101, *)      <= 20                    > 24         16
    TimesNet / GPT4TS  (128, 237)     100                      > 105        1
    TimesNet / GPT4TS  (128, 242)     100                      > 105        1
    TimesNet / GPT4TS  (128, 250)     100                      > 105        1
    TimesNet / GPT4TS  (60, 50)       48                       > 52         1
    TimesNet / GPT4TS  (176, 10)      17                       > 24         1
    TimesNet / GPT4TS  (152, 97)      120                      > +10%       1
    TimesNet / GPT4TS  (128, 36)      100                      > 112        1
    TimesNet / GPT4TS  (128, 28)      100                      > 112        1
    TimesNet / GPT4TS  (176, 12)      17                       > 36         1
    TimesNet / GPT4TS  (178, 8)       35                       > 42         1
    TimesNet / GPT4TS  (176, 3-13)    35                       > 42         1
    GPT4TS             (779, 111)     155                      > 192        1
    GPT4TS             (1488, 12-13)  1190                     > 1000       1
    GPT4TS             (*)            >= 1024                  > -10%       1


    Additional rules
    ----------------
    TimesNet / GPT4TS:
        If taux_nans > seq_len, seq_len = taux_nans + 2.

    GPT4TS:
        If seq_len >= 1024, seq_len is capped at 1020.
    """

    M = ts_m.shape[0]
    seq_len, batch_size = auto_seq_llms(data_x=ts_m, goal=strat, subset=True, high_limit=high, exception=True, verbose=verbose)

    if M < 500 and tr_ratio < 0.6 and security is not None: # more robust
        security = 1.15
    else:
        security = 1.05

    if consecutive:
        taux_nans = max_consecutive_nans_per_col_loop(ts_m).max(initial=0)
    else:
        taux_nans = int(np.isnan(ts_m).sum(axis=0).max())

    if M == 176 and taux_nans==10:
        return 32, 16, True, taux_nans, tr_ratio
    if M < 101 and taux_nans<=20:
        return 24, 16, True, taux_nans, tr_ratio

    if seq_len < taux_nans*security:
        limitation = int((ts_m.shape[0] * (1 - 0.05) * tr_ratio))
        if verbose:
            print(f"\n\n(INFO) Imputation adaptation with {model}, the number of NaNs values injected with ImputeGAP filled all the sequence, please change the seq_len or the contamination percentage. \n\tCurrently, the number of NaNs might reach {taux_nans} for a sequence and the size of seq_len is {seq_len}. Available data {limitation}\n")
        seq_len, batch_size = auto_seq_llms(data_x=ts_m, goal=strat, subset=True, high_limit=limitation, low_limit=int(taux_nans * security), exception=True, verbose=verbose)
        size_limitation = True

    if algo=="GPT4TS" or algo=="TimesNet":
        if taux_nans > seq_len or abs(taux_nans-seq_len) < 10:
            seq_len = max_consecutive_nans_per_col_loop(ts_m).max(initial=0)
            adding = int(seq_len*0.25)
            margin = int(ts_m.shape[1]*0.95)
            series = int(ts_m.shape[0]*0.95)
            batch_size = 8
            if margin < seq_len+adding or series < seq_len+adding:
                adding = int(seq_len*0.1)
                batch_size = 4
            if margin < seq_len+adding or series < seq_len+adding:
                adding = 4
                batch_size = 2
            if margin < seq_len+adding or series < seq_len+adding:
                adding = 2
                batch_size = 1
            seq_len = seq_len + adding

            if algo in ("TimesNet", "GPT4TS") and ts_m.shape[0] == 128 and (ts_m.shape[1] in [237, 242, 250]) and taux_nans == 100:
                seq_len = 105
                batch_size = 1
            if algo in ("TimesNet", "GPT4TS") and ts_m.shape[0] == 60 and ts_m.shape[1] ==50 and taux_nans == 48:
                seq_len = 52
                batch_size = 1
            if algo in ("TimesNet", "GPT4TS") and ts_m.shape[0] == 176 and ts_m.shape[1] == 10 and taux_nans == 17:
                seq_len = 24
                batch_size = 1
            if algo in ("TimesNet", "GPT4TS") and ts_m.shape[0] == 152 and ts_m.shape[1] ==97 and taux_nans == 120:
                adding = int(seq_len * 0.1)
                batch_size = 1
                seq_len = seq_len + adding
            if algo in ("TimesNet", "GPT4TS") and ts_m.shape[0] == 128 and ts_m.shape[1] == 36 and taux_nans == 100:
                batch_size = 1
                seq_len = 112
            if algo in ("TimesNet", "GPT4TS") and ts_m.shape[0] == 128 and ts_m.shape[1] == 28 and taux_nans == 100:
                batch_size = 1
                seq_len = 112
            if algo in ("TimesNet", "GPT4TS") and ts_m.shape[0] == 176 and ts_m.shape[1] == 12 and taux_nans == 17:
                batch_size = 1
                seq_len = 36
            if algo in ("TimesNet", "GPT4TS") and ts_m.shape[0] == 178 and ts_m.shape[1] == 8 and taux_nans == 35:
                batch_size = 1
                seq_len = 42
            if algo in ("TimesNet", "GPT4TS") and ts_m.shape[0] == 176 and ts_m.shape[1] in (3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 13) and taux_nans == 35:
                batch_size = 1
                seq_len = 42
            if algo in ("GPT4TS") and ts_m.shape[0] == 779 and ts_m.shape[1] in (111) and taux_nans == 155:
                batch_size = 1
                seq_len = 192
            if taux_nans > seq_len and algo in ("TimesNet", "GPT4TS"):
                seq_len = taux_nans + 2
            # if verbose:
            print(f"\t\t\t\tsequence adapted to {seq_len} for a number of NaNs of {taux_nans}")

        if algo in ("GPT4TS") and ts_m.shape[0] == 1488 and ts_m.shape[1] in (12, 13) and taux_nans == 1190:
            batch_size = 1
            seq_len = 1000
            size_limitation = True
            print(f"\t\t\t\tsequence adapted to {seq_len} for a number of NaNs of {taux_nans}")
        elif algo in ("GPT4TS") and taux_nans >= 1024:
            batch_size = 1
            seq_len = -10
            size_limitation = True
            print(f"\t\t\t\tsequence adapted to {seq_len} for a number of NaNs of {taux_nans}")

        elif algo in ("GPT4TS") and taux_nans >= 1024:
            batch_size = 1
            seq_len = -10
            size_limitation = True
            print(f"\t\t\t\tsequence adapted to {seq_len} for a number of NaNs of {taux_nans}")

        if algo in ("GPT4TS") and seq_len >= 1024:
            seq_len = 1020


    if verbose:
        print(f"\n\n(INFO) preprocessing of variables:\n\t{seq_len=}\n\t{batch_size=}\n\t{size_limitation=}\n\t{taux_nans=}\n")


    return seq_len, batch_size, size_limitation, taux_nans, tr_ratio


def patch_classifiers_stmvl(ts_m, window_size, gamma, alpha, verbose=False):
    """
    Adapt STMVL hyperparameters for specific dataset configurations.

    This function applies configuration patches for STMVL when particular dataset
    dimensions or missing-value patterns require adapted hyperparameters.

    The number of missing values is determined from the maximum number of
    consecutive NaNs found in any column of ``ts_m``.

    Parameters
    ----------
    ts_m : numpy.ndarray
        Two-dimensional time-series matrix containing the data to process.
        Missing values must be represented by ``NaN``.

    window_size : int
        Initial STMVL window size. This value may be replaced by a
        configuration-specific patch.

    gamma : float
        Initial STMVL gamma parameter. This value may be replaced by a
        configuration-specific patch.

    alpha : float
        Initial STMVL alpha parameter. This value may be replaced by a
        configuration-specific patch.

    verbose : bool, default=False
        Whether to print information when an STMVL patch is applied.

    Returns
    -------
    window_size : int
        Final STMVL window size after applying applicable patches.

    gamma : float
        Final STMVL gamma value after applying applicable patches.

    alpha : float
        Final STMVL alpha value after applying applicable patches.

    Specific configuration patches
    ------------------------------

    ALGO   SHAPE          CONTAMINATION / NaNs  > WINDOW_SIZE   GAMMA    ALPHA
    ----------------------------------------------------------------------------
    STMVL  (176, 3-13)    35                    > 96            0.5      1
    STMVL  (128, 3-15)*   any                   > 96            0.5      1
    STMVL  (128, 8/10)    any                   > 2             0.0001   1
    STMVL  (300, 20-39)   any                   > 4             0.001    1

    * For shape (128, 3-15), columns 8 and 10 are excluded from the first
      configuration and use the dedicated (128, 8/10) patch instead.

    Additional rules
    ----------------
    STMVL:
        If every column of ``ts_m`` contains at least one NaN, the missing values
        of the first column are replaced by the mean of the available values in
        that column.
    """

    taux_nans = max_consecutive_nans_per_col_loop(ts_m).max(initial=0)

    if ts_m.shape[0] == 176 and ts_m.shape[1] in (3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 13) and taux_nans == 35:
        window_size = 96
        gamma = 0.5
        alpha = 1

        if verbose:
            print(f"\n\tSTMVL PATCH USED FOR {ts_m.shape=} with following nans count of {taux_nans=}")
            print(f"\t\t\tnew vals: {window_size=} - {gamma=} - {alpha=}\n")

    if ts_m.shape[0] == 128 and ts_m.shape[1] in (3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15):
        window_size = 96
        gamma = 0.5
        alpha = 1
        if verbose:
            print(f"\n\tSTMVL PATCH USED FOR {ts_m.shape=} with following nans count of {taux_nans=}")
            print(f"\t\t\tnew vals: {window_size=} - {gamma=} - {alpha=}\n")

    if ts_m.shape[0] == 128 and ts_m.shape[1] in (8, 10):
        window_size = 2
        gamma = 0.0001
        alpha = 1
        if verbose:
            print(f"\n\tSTMVL PATCH USED FOR {ts_m.shape=} with following nans count of {taux_nans=}")
            print(f"\t\t\tnew vals: {window_size=} - {gamma=} - {alpha=}\n")

    if ts_m.shape[0] == 300 and ts_m.shape[1] in (20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39):
        window_size = 4
        gamma = 0.001
        alpha = 1
        if verbose:
            print(f"\n\tSTMVL PATCH USED FOR {ts_m.shape=} with following nans count of {taux_nans=}")
            print(f"\t\t\tnew vals: {window_size=} - {gamma=} - {alpha=}\n")



    if np.all(np.isnan(ts_m).any(axis=0)):
        ts_m[np.isnan(ts_m[:, 0]), 0] = np.nanmean(ts_m[:, 0])
        if verbose:
            print(f"\n\tSTMVL PATCH USED FOR {ts_m.shape=} with following nans count of {taux_nans=}")
            print(f"\t\t\tall cols had nans, patch effective for STMVL...\n")

    return window_size, gamma, alpha


def patch_classifiers_saits(incomp_data, seq_len, batch_size, sliding_windows, tr_ratio, verbose=True):
    """
    Adapt SAITS configuration for specific classification dataset configurations.

    This function applies configuration patches for SAITS when particular dataset
    dimensions and contamination levels require adapted sequence length, batch size,
    sliding-window configuration, or training ratio.

    The contamination level is determined from the total number of NaN values
    contained in ``incomp_data``.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        Two-dimensional contaminated time-series matrix. Missing values must be
        represented by ``NaN``.

    seq_len : int
        Initial sequence length. This value may be replaced by a
        configuration-specific patch.

    batch_size : int
        Initial batch size. This value may be replaced by a
        configuration-specific patch.

    sliding_windows : int
        Initial sliding-window configuration. This value may be replaced by a
        configuration-specific patch.

    tr_ratio : float
        Initial training ratio. This value may be replaced by a
        configuration-specific patch.

    verbose : bool, default=True
        Whether to print the total number of NaN values detected in the input data.

    Returns
    -------
    seq_len : int
        Final sequence length after applying applicable patches.

    batch_size : int
        Final batch size after applying applicable patches.

    sliding_windows : int
        Final sliding-window configuration after applying applicable patches.

    tr_ratio : float
        Final training ratio after applying applicable patches.

    Specific configuration patches
    ------------------------------

    ALGO   SHAPE       CONTAMINATION / NaNs  > SEQ_LEN  BATCH_SIZE  SLIDING  TR_RATIO
    ----------------------------------------------------------------------------------
    SAITS  (176, 4)    140                   > 40       16          1        0.50
    SAITS  (131, 4)    104                   > 36       16          1        0.33
    SAITS  (570, 4)    456                   > 128      16          1        0.50
    SAITS  (570, 8)    912                   > 128      16          1        0.40
    SAITS  (570, 5)    456                   > 128      16          1        0.40
    SAITS  (144, 13)   308                   > 20       16          1        0.40
    SAITS  (144, 16)   364                   > 16       16          1        0.40
    SAITS  (144, 18)   420                   > 20       16          1        0.40
    SAITS  (144, 9)    224                   > 16       16          1        0.40
    SAITS  (350, 3)    210                   > 72       16          1        0.40
    """

    total_nans = np.isnan(incomp_data).sum()
    if verbose:
        print("\n\t>>\tTotal NaNs:", total_nans, "\n")

    if incomp_data.shape[0] == 176 and incomp_data.shape[1] == 4 and total_nans == 140:
        seq_len = 40
        batch_size = 16
        sliding_windows = 1
        tr_ratio = 0.5

    if incomp_data.shape[0] == 131 and incomp_data.shape[1] == 4 and total_nans == 104:  # FacesUSR
        seq_len = 36
        batch_size = 16
        sliding_windows = 1
        tr_ratio = 0.33

    if incomp_data.shape[0] == 570 and incomp_data.shape[1] == 4 and total_nans == 456: # OliveOil
        seq_len = 128
        batch_size = 16
        sliding_windows = 1
        tr_ratio = 0.5

    if incomp_data.shape[0] == 570 and incomp_data.shape[1] == 8 and total_nans == 912: # OliveOil
        seq_len = 128
        batch_size = 16
        sliding_windows = 1
        tr_ratio = 0.4

    if incomp_data.shape[0] == 570 and incomp_data.shape[1] == 5 and total_nans == 456: # OliveOil
        seq_len = 128
        batch_size = 16
        sliding_windows = 1
        tr_ratio = 0.4

    if incomp_data.shape[0] == 144 and incomp_data.shape[1] == 13 and total_nans == 308: # Plane
        seq_len = 20
        batch_size = 16
        sliding_windows = 1
        tr_ratio = 0.4
    if incomp_data.shape[0] == 144 and incomp_data.shape[1] == 16 and total_nans == 364: # Plane
        seq_len = 16
        batch_size = 16
        sliding_windows = 1
        tr_ratio = 0.4
    if incomp_data.shape[0] == 144 and incomp_data.shape[1] == 18 and total_nans == 420: # Plane
        seq_len = 20
        batch_size = 16
        sliding_windows = 1
        tr_ratio = 0.4
    if incomp_data.shape[0] == 144 and incomp_data.shape[1] == 9 and total_nans == 224: # Plane
        seq_len = 16
        batch_size = 16
        sliding_windows = 1
        tr_ratio = 0.4
    if incomp_data.shape[0] == 350 and incomp_data.shape[1] == 3 and total_nans == 210: # Plane
        seq_len = 72
        batch_size = 16
        sliding_windows = 1
        tr_ratio = 0.4

    return seq_len, batch_size, sliding_windows, tr_ratio

def patch_classifiers_bitgraph(incomp_data, seq_len, kernel_set, kernel_size, subgraph_size, batch_size, verbose=True):
    """
    Adapt BITGraph configuration for specific classification dataset configurations.

    This function applies configuration patches for BITGraph when particular dataset
    dimensions and contamination levels require adapted sequence length, kernel
    configuration, subgraph size, or batch size.

    The contamination level is determined from the total number of NaN values
    contained in ``incomp_data``.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        Two-dimensional contaminated time-series matrix. Missing values must be
        represented by ``NaN``.

    seq_len : int
        Initial sequence length. This value may be replaced by a
        configuration-specific patch.

    kernel_set : list
        Initial set of kernels. This value may be replaced by a
        configuration-specific patch.

    kernel_size : int
        Initial kernel size. This value may be replaced by a
        configuration-specific patch.

    subgraph_size : int
        Initial subgraph size. This value may be replaced by a
        configuration-specific patch.

    batch_size : int
        Initial batch size. This value may be replaced by a
        configuration-specific patch.

    verbose : bool, default=True
        Whether to print the total number of NaN values detected in the input data.

    Returns
    -------
    seq_len : int
        Final sequence length after applying applicable patches.

    kernel_set : list
        Final kernel set after applying applicable patches.

    kernel_size : int
        Final kernel size after applying applicable patches.

    subgraph_size : int
        Final subgraph size after applying applicable patches.

    batch_size : int
        Final batch size after applying applicable patches.

    Specific configuration patches
    ------------------------------

    ALGO      SHAPE      CONTAMINATION / NaNs  > SEQ_LEN  KERNEL_SET  KERNEL_SIZE  SUBGRAPH_SIZE  BATCH_SIZE
    ---------------------------------------------------------------------------------------------------------
    BITGraph  (350, 3)   70 or 210             > 72       [1, 2]      2            2              2
    BITGraph  (570, 4)   456                   > 128      [1, 2]      2            2              2
    """

    total_nans = np.isnan(incomp_data).sum()
    if verbose:
        print("\n\t>>\tTotal NaNs:", total_nans, "\n")

    if incomp_data.shape[0] == 350 and incomp_data.shape[1] == 3 and (total_nans == 70 or total_nans == 210): # FaceFour
        seq_len = 72
        kernel_set = [1,2]
        kernel_size = 2
        subgraph_size = 2
        batch_size = 2
    if incomp_data.shape[0] == 570 and incomp_data.shape[1] == 4 and (total_nans == 456): # OliveOil
        seq_len = 128
        kernel_set = [1,2]
        kernel_size = 2
        subgraph_size = 2
        batch_size = 2

    return seq_len, kernel_set, kernel_size, subgraph_size, batch_size


def patch_classifiers_grin(incomp_data, seq_len, batch_size, verbose=True):
    """
    Adapt GRIN configuration for small classification datasets.

    This function applies a configuration patch for GRIN when the time-series
    length is small and requires an adapted sequence length and batch size.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        Two-dimensional contaminated time-series matrix. Missing values must be
        represented by ``NaN``.

    seq_len : int
        Initial sequence length. This value may be replaced by the GRIN patch.

    batch_size : int
        Initial batch size. This value may be replaced by the GRIN patch.

    verbose : bool, default=True
        Whether to print the total number of NaN values detected in the input data.

    Returns
    -------
    seq_len : int
        Final sequence length after applying applicable patches.

    batch_size : int
        Final batch size after applying applicable patches.


    Specific configuration patches
    ------------------------------

    ALGO   SHAPE       CONTAMINATION / NaNs  > SEQ_LEN  BATCH_SIZE
    ---------------------------------------------------------------
    GRIN   (<=80,all)   any                   > 16       16
    """
    total_nans = np.isnan(incomp_data).sum()
    if verbose:
        print("\n\t>>\tTotal NaNs:", total_nans, "\n")

    if incomp_data.shape[0] <=80: # *
        seq_len = 16
        batch_size = 16

    return seq_len, batch_size


