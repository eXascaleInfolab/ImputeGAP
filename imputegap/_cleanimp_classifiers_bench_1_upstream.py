from imputegap.recovery.benchmark import Benchmark
from imputegap.tools import utils

# launch the evaluation
bench = Benchmark()
bench.eval_downstream_classification(classifiers=["arsenal"], # for all: utils.list_of_classifiers(),
                                     algorithms=["MeanImpute", "TRMF"], # for all: utils.list_of_algorithms()
                                     datasets=["Car", "Computers", "Wine"], # for all: utils.get_datasets_classifiers(),
                                     patterns=["mcar", "aligned_series", "aligned_timestamps"],
                                     x_axis=[0.1, 0.2, 0.4, 0.6, 0.8],
                                     metrics=["RMSE", "MAE", "MI", "CORRELATION", "RUNTIME"],
                                     normalizer="z-score",
                                     report_title="cleanimp_benchmark_cl_up",
                                     contamination_by_class=True,
                                     imputation_by_class=True,
                                     bypass_error=False,
                                     evaluate_upstream=True,
                                     to_cache=True,
                                     use_cache=False,
                                     run_downstream=False,
                                     fixed_rate=0.2,
                                     inner_plots=False,
                                     referential=False,
                                     plots=False,
                                     generate_plot=True,
                                     verbose=False)