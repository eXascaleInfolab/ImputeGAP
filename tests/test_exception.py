import unittest

import numpy as np
import pytest

from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils


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
        utils.save_optimization(optimal_params=(1,0.1,10), algorithm="cdrec", dataset="eeg", optimizer="b")
        utils.save_optimization(optimal_params=(24,50,64,1,108,0.3,0), algorithm="mrnn", dataset="eeg", optimizer="b")
        utils.save_optimization(optimal_params=(1,0.1,10), algorithm="stmvl", dataset="eeg", optimizer="b")
        utils.save_optimization(optimal_params=(1, ""), algorithm="iim", dataset="eeg", optimizer="b")
        test = True
        assert test is not None
