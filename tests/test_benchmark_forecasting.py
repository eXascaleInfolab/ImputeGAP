import math
import unittest

from imputegap.recovery.benchmark import Benchmark

class TestBenchmarkForecasting(unittest.TestCase):

    def assert_valid_metrics(self, scores, expected_metrics):
        """
        Check that expected metrics:
        - exist
        - are numeric
        - are finite (not NaN / inf)
        - have sensible values
        """

        for metric in expected_metrics:

            self.assertIn(metric, scores, f"{metric} is missing from scores: {scores}")
            value = scores[metric]

            self.assertIsInstance(value, (int, float), f"{metric} should be numeric, got {type(value)}: {value}")

            self.assertFalse(math.isnan(value), f"{metric} is NaN: {scores}")

            self.assertFalse(math.isinf(value), f"{metric} is infinite: {scores}")

            if metric in ["RMSE", "RUNTIME", "DOWNSTREAM_SMAPE", ]:
                self.assertGreaterEqual(value, 0, f"{metric} should be >= 0, got {value}")

            if metric == "CORRELATION":
                self.assertGreaterEqual(value, -1, f"CORRELATION < -1: {value}" )
                self.assertLessEqual(value, 1, f"CORRELATION > 1: {value}")


    def test_forecasting_upstream_mcar(self):
        bench = Benchmark()

        bench.eval_downstream_forecasting(
            horizon=12,
            algorithms=["cdrec"],
            datasets=["paris"],
            patterns=["mcar"],
            x_axis=[0.2],
            metrics=["RMSE", "MI", "CORRELATION", "RUNTIME"],
            normalizer="z_score",
            to_cache=False,
            use_cache=False,
            run_downstream=False,
            evaluate_upstream=True,
            fixed_rate=0.2,
            generate_plot=False,
            plots=False,
            inner_plots=False,
            verbose=False,
            nbr_series=5,
            nbr_vals=100, )

        scores = (bench.list_results[0]["paris"]["mcar"]["cdrec"]["no-model"]["0.2"]["scores"])

        self.assert_valid_metrics(scores, ["RMSE", "MI", "CORRELATION", "RUNTIME"])


    def test_forecasting_upstream_aligned_series(self):
        bench = Benchmark()

        bench.eval_downstream_forecasting(
            horizon=12,
            algorithms=["cdrec"],
            datasets=["paris"],
            patterns=["aligned_series"],
            x_axis=[0.2],
            metrics=["RMSE", "MI", "CORRELATION", "RUNTIME"],
            normalizer="z_score",
            to_cache=False,
            use_cache=False,
            run_downstream=False,
            evaluate_upstream=True,
            fixed_rate=0.2,
            generate_plot=False,
            plots=False,
            inner_plots=False,
            verbose=False,
            nbr_series=5,
            nbr_vals=100,
        )

        scores = (bench.list_results[0]["paris"]["aligned_series"]["cdrec"]["no-model"]["0.2"]["scores"])

        self.assert_valid_metrics(scores, ["RMSE", "MI", "CORRELATION", "RUNTIME"])


    def test_forecasting_upstream_aligned_timestamps(self):
        bench = Benchmark()

        bench.eval_downstream_forecasting(
            horizon=12,
            algorithms=["cdrec"],
            datasets=["paris"],
            patterns=["aligned_timestamps"],
            x_axis=[0.2],
            metrics=["RMSE", "MI", "CORRELATION", "RUNTIME"],
            normalizer="z_score",
            to_cache=False,
            use_cache=False,
            run_downstream=False,
            evaluate_upstream=True,
            fixed_rate=0.2,
            generate_plot=False,
            plots=False,
            inner_plots=False,
            verbose=False,
            nbr_series=5,
            nbr_vals=100,
        )

        scores = (bench.list_results[0]["paris"]["aligned_timestamps"]["cdrec"]["no-model"]["0.2"]["scores"])

        self.assert_valid_metrics(scores,["RMSE", "MI", "CORRELATION", "RUNTIME"])


    def test_forecasting_multiple_rates(self):
        bench = Benchmark()

        bench.eval_downstream_forecasting(
            horizon=12,
            algorithms=["cdrec"],
            datasets=["paris"],
            patterns=["mcar"],
            x_axis=[0.1, 0.2, 0.4],
            metrics=["RMSE", "MI", "CORRELATION", "RUNTIME"],
            normalizer="z_score",
            to_cache=False,
            use_cache=False,
            run_downstream=False,
            evaluate_upstream=True,
            fixed_rate=0.2,
            generate_plot=False,
            plots=False,
            inner_plots=False,
            verbose=False,
            nbr_series=5,
            nbr_vals=100,
        )

        rates = (bench.list_results[0]["paris"]["mcar"]["cdrec"]["no-model"])

        for rate in ["0.1", "0.2", "0.4"]:
            self.assertIn(rate, rates)
            scores = rates[rate]["scores"]
            self.assert_valid_metrics(scores,["RMSE", "MI", "CORRELATION", "RUNTIME"])


    def test_forecasting_downstream(self):
        bench = Benchmark()

        bench.eval_downstream_forecasting(
            forecasters=["exp-smoothing"],
            horizon=12,
            algorithms=["cdrec"],
            datasets=["paris"],
            patterns=["mcar"],
            x_axis=[0.2],
            normalizer="z_score",
            to_cache=False,
            use_cache=False,
            run_downstream=True,
            evaluate_upstream=False,
            fixed_rate=0.2,
            generate_plot=False,
            plots=False,
            inner_plots=False,
            referential=True,
            verbose=False,
            nbr_series=20,
            nbr_vals=120,
        )

        scores = (bench.list_results[0]["paris"]["mcar"]["cdrec"]["exp-smoothing"]["0.2"]["scores"])

        print(f"test {scores=}")

        for metric, value in scores.items():

            self.assertIsInstance(value,(int, float),f"{metric} is not numeric: {value}")

            self.assertFalse(math.isnan(value),f"{metric} is NaN: {scores}")

            self.assertFalse(math.isinf(value),f"{metric} is infinite: {scores}")

        self.assert_valid_metrics(
            scores,
            [
                "mse_groundtruth",
                "mse_imputer",
                "mse_baseline",
                "mae_groundtruth",
                "mae_imputer",
                "mae_baseline",
                "smape_groundtruth",
                "smape_imputer",
                "smape_baseline",
            ]
        )


if __name__ == "__main__":
    unittest.main()
