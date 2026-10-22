from imputegap.recovery.benchmark import Benchmark
from imputegap.tools import utils

x = ["PriSTI"]
ps = ["aligned_timestamps"]
points = [0.1, 0.2, 0.4, 0.6, 0.8]
# launch the evaluation

for alg in x:
    for p in ps:
        for point in points:
            bench = Benchmark()
            bench.eval_downstream_classification(classifiers=["arsenal"],
                                                 algorithms=[alg],
                                                 datasets=utils.get_datasets_classifiers(),
                                                 patterns=[p],
                                                 x_axis=[point],
                                                 metrics=["RMSE", "DOWNSTREAM_ACC", "MI", "CORRELATION", "RUNTIME"],
                                                 normalizer="z-score",
                                                 report_title="opti_rerun_cl",
                                                 contamination_by_class=True,
                                                 imputation_by_class=True,
                                                 bypass_error=False,
                                                 evaluate_upstream=True,
                                                 to_cache=True,
                                                 use_cache=True,
                                                 run_downstream=False,
                                                 fixed_rate=0.2,
                                                 verbose=False,
                                                 inner_plots=False,
                                                 referential=False,
                                                 plots=False,
                                                 generate_plot=False)