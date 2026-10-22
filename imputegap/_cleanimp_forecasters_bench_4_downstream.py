from imputegap.recovery.benchmark import Benchmark
from imputegap.tools import utils

# launch the evaluation
bench = Benchmark()
bench.eval_downstream_forecasting(forecasters=["chronos"],
                                  horizon=12,
                                  algorithms=["MeanImpute", "MICE"], # utils.list_of_top_cleanimp_for_algorithms(),
                                  datasets=["ili", "paris"], # utils.get_dataset_forecasters(directory='datasets/forecast/'),
                                  patterns=["mcar", "seqn"],
                                  x_axis=[0.2, 0.4, 0.6],
                                  metrics=["RMSE", "DOWNSTREAM_SMAPE", "MI", "CORRELATION", "RUNTIME"],
                                  normalizer="z-score",
                                  report_title="cleanimp_benchmark_for_down",
                                  bypass_error=False,
                                  to_cache=True,
                                  use_cache=False,
                                  run_downstream=True,
                                  evaluate_upstream=False,
                                  fixed_rate=0.2,
                                  verbose=False,
                                  inner_plots=False,
                                  plots=False,
                                  generate_plot=True,
                                  referential=False,
                                  nbr_series=10000,
                                  nbr_vals=10000)