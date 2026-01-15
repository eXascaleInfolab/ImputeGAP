from imputegap.recovery.benchmark import Benchmark
from imputegap.tools import utils

my_algorithms = ["MeanImpute", "SoftImpute"]

my_opt = "default_params"

my_datasets = ["eeg-alcohol"]

my_patterns = ["mcar"]

range = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

# launch the evaluation
bench = Benchmark()
bench.eval(algorithms=my_algorithms, datasets=my_datasets, patterns=my_patterns, x_axis=range, optimizer=my_opt)