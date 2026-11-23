from imputegap.recovery.benchmark import Benchmark
from imputegap.tools import utils

datasets = utils.get_datasets_classifiers()

# launch the evaluation
bench = Benchmark()
bench.eval_downstream_classification(classifiers=["arsenal"],
                                     algorithms=["TRMF"],
                                     datasets=utils.get_datasets_classifiers(),
                                     patterns=["mcar", "aligned_series", "aligned_timestamps"],
                                     x_axis=[0.1, 0.2, 0.4, 0.6, 0.8],
                                     metrics=["RMSE", "DOWNSTREAM_ACC", "MI", "CORRELATION", "RUNTIME"],
                                     normalizer="z-score",
                                     report_title="cardinality_T_F_trmf",
                                     contamination_by_class=True,
                                     imputation_by_class=False,
                                     bypass_error=False,
                                     to_cache=False,
                                     use_cache=False,
                                     run_downstream=True,
                                     referential=False,
                                     evaluate_upstream=False,
                                     fixed_rate=0.2,
                                     verbose=False,
                                     plots=False,
                                     generate_plot=False)