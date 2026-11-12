# ImputeGAP import
from imputegap.recovery.benchmark import BenchGAP

# launch the evaluation
bench = BenchGAP()
bench.eval_imputation(algorithms=["MeanImpute", "SoftImpute"],
           datasets=["eeg-alcohol"],
           patterns=["mcar"],
           x_axis=[0.1, 0.2, 0.4, 0.6, 0.8],
           use_cache=False)

# display the plot
bench.subplots.show()