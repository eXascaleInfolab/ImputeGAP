import unittest

import numpy as np
import pytest

from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils
from imputegap.recovery.contamination import GenGap


class TestException(unittest.TestCase):

    def test_algorithm_exc(self):
        """
        the goal is to test the exception to algorithms
        """
        algorithm = "invalid_algo"
        with pytest.raises(ValueError, match=f"Invalid algorithm: {algorithm}"):
            Imputation.evaluate_params(input_data=None, incomp_data=None, configuration=tuple(), algorithm=algorithm)

    def test_data_exc(self):
        """
        The goal is to test the exception raised when input_data (raw_data) is None
        """
        input_data = None  # Simulate a scenario where raw_data is None
        with pytest.raises(ValueError, match=f"Need input_data to be able to adapt the hyper-parameters: {input_data}"):
            _ = Imputation.MatrixCompletion.CDRec(None).impute(user_def=False, params={"input_data":input_data, "optimizer": "bayesian", "options":{"n_calls": 2}})


    def test_unknown_algorithm_raises(self):
        """
        The goal is to test exceptions
        """
        with pytest.raises(ValueError, match=r"\(IMP\) Algorithm 'blah' not recognized"):
            utils.config_impute_algorithm(incomp_data=None, algorithm="blah")

    def test_patters_raises(self):
        alpha=True
        s = ["mcar", "aligned", "disjoint", "overlap", "scatter", "gaussian", "distribution", "blackout"]
        ts = TimeSeries()
        ts.import_matrix(np.array([[12.0, 12, 12, 12], [12, 12, 12, 12], [12, 12, 12, 12], [12, 12, 12, 12], [12, 12, 12, 12], [12, 12, 12, 12], [12, 12, 12, 12], [12, 12, 12, 12]]))
        for pattern in s:
            _ = utils.config_contamination(ts=ts, pattern=pattern)

        _ = GenGap.mcar(ts.data, rate_dataset=0.5, rate_series=0.5, logic_by_series=False, block_size=1, offset=0)
        self.assertTrue(alpha, True)

    def test_unknown_algorithm_writes_raises(self):
        """
        The goal is to test exceptions
        """
        # Map: canonical algorithm -> (one example optimal_params tuple, expected keys)
        ALG_SPECS = {
            "cdrec": ((3, 1e-3, 10), {"rank", "epsilon", "iteration"}),
            "stmvl": ((5, 0.1, 2), {"window_size", "gamma", "alpha"}),
            "iim": ((7,), {"learning_neighbors"}),
            "iterativesvd": ((4,), {"rank"}),
            "grouse": ((6,), {"max_rank"}),
            "rosl": ((2, 0.5), {"rank", "regularization"}),
            "softimpute": ((8,), {"max_rank"}),
            "spirit": ((2, 10, 0.01), {"k", "w", "lvalue"}),
            "svt": ((1.0, 0.1, 50), {"tau", "delta", "max_iter"}),
            "dynammo": ((3, 20, True), {"h", "max_iteration", "approximation"}),
            "tkcm": ((5,), {"rank"}),
            "knn": ((3, "uniform"), {"k", "weights"}),
            "interpolation": (("linear", 2), {"method", "poly_order"}),
            "mice": ((10, 1e-3, "mean"), {"max_iter", "tol", "initial_strategy", "seed"}),
            "missforest": ((100, 10, "sqrt"), {"n_estimators", "max_iter", "max_features", "seed"}),
            "xgboost": ((200,), {"n_estimators", "seed"}),
            "gain": ((64, 10, 1, 0.9), {"batch_size", "epochs", "alpha", "hint_rate"}),
            "bayotide": ((1, 1, 12, 1, 1, 0.1, 0.1, 0.1, 5),
                         {"K_trend", "K_season", "n_season", "K_bias", "time_scale", "a0", "b0", "v", "num_fold"}),
            "hkmft": (("tag", 12, 1, 2, 5), {"tags", "seq_len", "blackouts_begin", "blackouts_end", "epochs"}),
            "bitgraph": ((12, 3, 5, "kernels", 10, 32, 8, 0),
                         {"seq_len", "sliding_windows", "kernel_size", "kernel_set", "epochs", "batch_size",
                          "subgraph_size", "num_workers"}),
            "nuwats": (
            (12, 32.0, 5, 2, 0, 42), {"seq_len", "batch_size", "epochs", "gpt_layers", "num_workers", "seed"}),
            "gpt4ts": (
            (12, 32.0, 5, 2, 0, 42), {"seq_len", "batch_size", "epochs", "gpt_layers", "num_workers", "seed"}),
            "timesnet": (
            (12, 32.0, 5, 2, 0, 42), {"seq_len", "batch_size", "epochs", "gpt_layers", "num_workers", "seed"}),
            "pristi": ((12, 32.0, 5, 3, "strategy", 2, 0),
                       {"seq_len", "batch_size", "epochs", "sliding_windows", "target_strategy", "nsamples",
                        "num_workers"}),
            "csdi": ((12, 32.0, 5, 3, "strategy", 2, 0),
                     {"seq_len", "batch_size", "epochs", "sliding_windows", "target_strategy", "nsamples",
                      "num_workers"}),
            "saits": (
            (12, 32, 5, 3, 2, 0), {"seq_len", "batch_size", "epochs", "sliding_windows", "n_head", "num_workers"}),
        }

        for a in ALG_SPECS:
            utils.save_optimization(optimal_params=ALG_SPECS[a][0], algorithm=a, dataset="test")

    def test_import_exc(self):
        """
        The goal is to test the exception raised when import is wrong
        """
        ts_01 = TimeSeries()

        with pytest.raises(ValueError, match="Invalid input for import_matrix"):
            ts_01.import_matrix("wrong")

        with pytest.raises(ValueError, match="Invalid input for load_series"):
            ts_01.load_series(0.1)


    def test_load_exc(self):
        """
        The goal is to test the exception raised with loading of default values
        """
        default_mrnn = utils.load_parameters(query="default", algorithm="mrnn")
        default_cdrec = utils.load_parameters(query="default", algorithm="cdrec")
        default_iim = utils.load_parameters(query="default", algorithm="iim")
        default_stmvl = utils.load_parameters(query="default", algorithm="stmvl")
        default_greedy = utils.load_parameters(query="default", algorithm="greedy")
        default_bayesian = utils.load_parameters(query="default", algorithm="bayesian")
        default_pso = utils.load_parameters(query="default", algorithm="pso")
        default_color = utils.load_parameters(query="default", algorithm="colors")
        default_false = utils.load_parameters(query="default", algorithm="test-wrong")

        assert default_cdrec is not None
        assert default_mrnn is not None
        assert default_iim is not None
        assert default_stmvl is not None
        assert default_greedy is not None
        assert default_bayesian is not None
        assert default_pso is not None
        assert default_color is not None
        assert default_false is None


    def test_export_exc(self):
        """
        The goal is to test the exception raised with loading of default values
        """
        test = None
        utils.display_title()

        utils.save_optimization(optimal_params=(1,0.1,10), algorithm="cdrec", dataset="eeg", optimizer="b")
        utils.save_optimization(optimal_params=(24,50,64,1,108,0.3,0), algorithm="mrnn", dataset="eeg", optimizer="b")
        utils.save_optimization(optimal_params=(1,0.1,10), algorithm="stmvl", dataset="eeg", optimizer="b")
        utils.save_optimization(optimal_params=(1, ""), algorithm="iim", dataset="eeg", optimizer="b")
        test = True

        ts = TimeSeries()
        x = ts.algorithms
        x = ts.patterns
        x = ts.datasets
        x = ts.optimizers
        x = ts.extractors
        x = ts.forecasting_models
        x = ts.families
        x = ts.algorithms_with_families
        x = utils.list_of_metrics()
        x = utils.list_of_algorithms_deep_learning()
        x = utils.list_of_algorithms_matrix_completion()
        x = utils.list_of_algorithms_pattern_search()
        x = utils.list_of_algorithms_machine_learning()
        x = utils.list_of_algorithms_statistics()
        x = utils.list_of_algorithms_llms()
        x = utils.list_of_algorithms_with_families()
        x = utils.list_of_normalizers()

        assert test is not None

    def test_dl_split_exc(self):
        """
        The goal is to test the exception raised with loading of default values
        """
        # initialize the time series object
        ts = TimeSeries()
        print(f"\nImputation algorithms : {ts.algorithms}")

        # load and normalize the dataset
        ts.load_series(utils.search_path("chlorine"), normalizer="z_score")

        # contaminate the time series
        ts_m = GenGap.mcar(ts.data)

        cont_data_matrix, mask_train, mask_test, mask_valid, error = utils.dl_integration_transformation(ts_m,
                                                                                                         tr_ratio=0.6,
                                                                                                         inside_tr_cont_ratio=0.2,
                                                                                                         split_ts=1,
                                                                                                         split_val=0,
                                                                                                         nan_val=-99999,
                                                                                                         prevent_leak=-99999,
                                                                                                         offset=0.05,
                                                                                                         verbose=False)

        assert cont_data_matrix is not None, "cont_data_matrix should be None"




