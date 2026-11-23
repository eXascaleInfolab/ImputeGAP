from imputegap.recovery.benchmark import Benchmark
from imputegap.tools import utils

for xs in sorted([
        "arima",
    ]):

    # launch the evaluation
    bench = Benchmark()
    bench.eval_downstream_forecasting(forecasters=[xs],
                                      horizons=[12],
                                      algorithms=["MeanImpute"],
                                      datasets=utils.get_dataset_forecasters(directory='datasets/forecast/'),
                                      patterns=["aligned_series"],
                                      x_axis=[0.1],
                                      metrics=["RMSE", "DOWNSTREAM_SMAPE", "MI", "CORRELATION", "RUNTIME"],
                                      normalizer="z-score",
                                      report_title="raw",
                                      bypass_error=False,
                                      to_cache=True,
                                      use_cache=False,
                                      run_downstream=True,
                                      evaluate_upstream=False,
                                      fixed_rate=0.2,
                                      verbose=False,
                                      inner_plots=False,
                                      plots=False,
                                      generate_plot=False,
                                      referential=True,
                                      nbr_series=10000,
                                      nbr_vals=10000)