from imputegap.recovery.benchmark import Benchmark
from imputegap.tools import utils

# launch the evaluation
bench = Benchmark()
bench.eval_downstream_forecasting(forecasters=["chronos"],
                                  horizons=[12],
                                  algorithms=["SAITS"],
                                  datasets=utils.get_dataset_forecasters(directory='datasets/forecast/'),
                                  patterns=["aligned_series"],
                                  x_axis=[0.1, 0.2, 0.4, 0.6, 0.8],
                                  metrics=["RMSE", "DOWNSTREAM_SMAPE", "MI", "CORRELATION", "RUNTIME"],
                                  normalizer="z-score",
                                  report_title="chronos_aligned_series",
                                  bypass_error=False,
                                  to_cache=True,
                                  use_cache=True,
                                  run_downstream=True,
                                  evaluate_upstream=False,
                                  fixed_rate=0.2,
                                  verbose=False,
                                  inner_plots=False,
                                  plots=False,
                                  generate_plot=False,
                                  referential=False,
                                  nbr_series=10000,
                                  nbr_vals=10000, )