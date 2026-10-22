from imputegap.recovery.benchmark import Benchmark
from imputegap.tools import utils

# launch the evaluation
bench = Benchmark()
bench.eval_downstream_classification(classifiers=["arsenal"],
                                     algorithms=utils.list_of_top_cleanimp_cl_algorithms(),
                                     datasets=["MoteStrain"],
                                     patterns=["mcar"],
                                     x_axis=[0.1, 0.2, 0.4, 0.6, 0.8],
                                     metrics=["RMSE", "DOWNSTREAM_ACC", "MI", "CORRELATION", "RUNTIME"],
                                     normalizer="z-score",
                                     report_title="cardinality_T_F",
                                     contamination_by_class=True,
                                     imputation_by_class=False,
                                     bypass_error=False,
                                     to_cache=True,
                                     use_cache=True,
                                     run_downstream=True,
                                     referential=False,
                                     evaluate_upstream=False,
                                     fixed_rate=0.2,
                                     verbose=False,
                                     plots=False,
                                     generate_plot=False)