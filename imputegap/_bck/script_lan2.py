from imputegap.recovery.benchmark import Benchmark
from imputegap.tools import utils

#tmp = utils.get_datasets_classifiers()
#remove {"ACSF1", "ChlorineConcentration", "FaceAll", "InsectWingbeatSound", "Plane"}, from the list
tmp = utils.get_datasets_classifiers()
# Remove unwanted datasets
tmp = [
    dataset for dataset in tmp
    if dataset not in {
        "ACSF1",
        "ChlorineConcentration",
        "FaceAll",
        "InsectWingbeatSound",
        "Plane",
    }
]

c = ["arsenal", "catch22", "cboss", "cif", "cnn", "itde"]
b = ["knn", "lstm", "proxstump", "shapedtw", "signature"]
a = ["stc", "svc", "tsf", "tsfresh", "weasel"]


# launch the evaluation
bench = Benchmark()
bench.eval_downstream_classification(classifiers=a,
                                     algorithms=utils.list_of_top_cleanimp_cl_algorithms(),
                                     datasets=utils.get_datasets_classifiers(),
                                     patterns=["mcar", "aligned_series", "aligned_timestamps"],
                                     x_axis=[0.1, 0.2, 0.4, 0.6, 0.8],
                                     metrics=["RMSE", "DOWNSTREAM_ACC", "MI", "CORRELATION", "RUNTIME"],
                                     normalizer="z-score",
                                     report_title="cleanimp_benchmark_cl_down",
                                     contamination_by_class=True,
                                     imputation_by_class=True,
                                     bypass_error=True,
                                     to_cache=True,
                                     use_cache=True,
                                     run_downstream=True,
                                     referential=False,
                                     evaluate_upstream=False,
                                     fixed_rate=0.2,
                                     verbose=False,
                                     plots=False,
                                     generate_plot=False)