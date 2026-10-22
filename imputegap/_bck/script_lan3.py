from imputegap.recovery.benchmark import Benchmark
from imputegap.tools import utils

# launch the evaluation
bench = Benchmark()
bench.eval_downstream_classification(classifiers=["cif"],
                                     algorithms=["MeanImpute"],
                                     datasets=["Beef"],
                                     patterns=["mcar", "aligned_series", "aligned_timestamps"],
                                     x_axis=[0.1, 0.2, 0.4, 0.6, 0.8],
                                     metrics=["RMSE", "DOWNSTREAM_ACC", "MI", "CORRELATION", "RUNTIME"],
                                     normalizer="z-score",
                                     report_title="cif",
                                     contamination_by_class=True,
                                     imputation_by_class=True,
                                     bypass_error=False,
                                     evaluate_upstream=False,
                                     to_cache=False,
                                     use_cache=False,
                                     run_downstream=True,
                                     fixed_rate=0.2,
                                     verbose=False,
                                     inner_plots=False,
                                     referential=False,
                                     plots=False,
                                     generate_plot=False)