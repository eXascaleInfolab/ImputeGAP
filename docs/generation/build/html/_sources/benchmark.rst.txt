=========
Benchmark
=========


ImputeGAP can serve as a common test-bed for comparing the effectiveness and efficiency of time series imputation algorithms [33]_.  Users have full control over the benchmark by customizing various parameters, including the list of the algorithms to compare, the optimizer, the datasets to evaluate, the missingness patterns, the range of missing values, and the performance metrics.


The benchmarking module can be utilized as follows:

.. code-block:: python

    from imputegap.recovery.benchmark import Benchmark

    my_algorithms = ["SoftImpute", "KNNImpute"]

    my_opt = ["default_params"]

    my_datasets = ["eeg-alcohol"]

    my_patterns = ["mcar"]

    range = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

    my_metrics = ["*"]

    # launch the evaluation
    bench = Benchmark()
    bench.eval(algorithms=my_algorithms, datasets=my_datasets, patterns=my_patterns, x_axis=range, metrics=my_metrics, optimizers=my_opt)





You can enable the optimizer using the following command:

.. code-block:: python

    opt = {"optimizer": "ray_tune", "options": {"n_calls": 1, "max_concurrent_trials": 1}}
    my_opt = [opt]


.. [33] Mourad Khayati, Alberto Lerner, Zakhar Tymchenko, Philippe Cudré-Mauroux: Mind the Gap: An Experimental Evaluation of Imputation of Missing Values Techniques in Time Series. Proc. VLDB Endow. 13(5): 768-782 (2020)


.. raw:: html

   <br>