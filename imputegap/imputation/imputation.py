import os
import toml
from imputegap.algorithms.cdrec import native_cdrec_param
from imputegap.algorithms.zero_impute import zero_impute
from imputegap.evaluation.evaluation import Evaluation


class Imputation:

    def load_toml(filepath = "../env/default_values.toml"):
        """
        Load default values of algorithms
        :return: the config of default values
        """
        if not os.path.exists(filepath):
            filepath = "./env/default_values.toml"

        with open(filepath, "r") as file:
            config = toml.load(file)

        return config

    class MR:
        def cdrec(ground_truth, contamination, params=None):
            """
            Imputation of data with CDREC algorithm
            @author Quentin Nater

            :param ground_truth: original time series without contamination
            :param contamination: time series with contamination
            :param params: [Optional] parameters of the algorithm, if None, default ones are loaded

            :return: imputed_matrix, metrics : all time series with imputation data and their metrics
            """

            if params is not None:
                truncation_rank, epsilon, iterations = params
            else:
                config = Imputation.load_toml()
                truncation_rank = config['cdrec']['default_reduction_rank']
                epsilon = config['cdrec']['default_epsilon_str']
                iterations = config['cdrec']['default_iteration']

            imputed_matrix = native_cdrec_param(__py_matrix=contamination, __py_rank=int(truncation_rank),
                                                __py_eps=float("1" + epsilon), __py_iters=int(iterations))

            metrics = Evaluation(ground_truth, ground_truth + 0.1, contamination).metrics_computation()

            print("CDREC Imputation completed without error.\n")

            return imputed_matrix, metrics

    class Stats:
        def zero_impute(ground_truth, contamination, params=None):
            imputed_matrix = zero_impute(ground_truth, contamination, params)
            metrics = Evaluation(ground_truth, imputed_matrix, contamination).metrics_computation()

            print("ZERO Imputation completed without error.\n")

            return imputed_matrix, metrics