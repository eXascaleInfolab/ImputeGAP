from imputegap.recovery.benchmark import Benchmark
from imputegap.tools import utils

# launch the evaluation
bench = Benchmark()
bench.eval_downstream_classification(classifiers=["arsenal"], # for all: utils.list_of_classifiers(),
                                     algorithms=["MeanImpute", "TRMF"], # for all: utils.list_of_top_cleanimp_cl_algorithms(),
                                     datasets=["Car", "Computers", "Wine"], # utils.get_datasets_classifiers(),
                                     patterns=["mcar", "aligned_series", "aligned_timestamps"],
                                     x_axis=[0.1, 0.2, 0.4, 0.6, 0.8],
                                     metrics=["RMSE", "DOWNSTREAM_ACC", "MI", "CORRELATION", "RUNTIME"],
                                     normalizer="z-score",
                                     report_title="cleanimp_benchmark_cl_down",
                                     contamination_by_class=True,
                                     imputation_by_class=True,
                                     bypass_error=True,
                                     to_cache=True,
                                     use_cache=False,
                                     run_downstream=True,
                                     referential=False,
                                     evaluate_upstream=False,
                                     fixed_rate=0.2,
                                     verbose=False,
                                     plots=False,
                                     generate_plot=False)