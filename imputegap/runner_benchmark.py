# ImputeGAP import
from imputegap.recovery.benchmark import Benchmark

# launch the evaluation
bench = Benchmark()
bench.eval_imputation(algorithms=["MeanImpute", "SoftImpute"],
           datasets=["eeg-alcohol"],
           patterns=["mcar"],
           x_axis=[0.1, 0.2, 0.4, 0.6, 0.8])

# display the plot
bench.subplots.show()