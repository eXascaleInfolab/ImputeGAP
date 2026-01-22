=========
Tutorials
=========


.. _loading:

Dataset Loading
---------------

ImputeGAP comes with several time series datasets. The datasets are listed `here <datasets.html>`_.

As an example, we use the eeg-alcohol dataset, composed of individuals with a genetic predisposition to alcoholism. The dataset contains measurements from 64 electrodes placed on subject’s scalps, sampled at 256 Hz. The dimensions of the dataset are 64 series, each containing 256 values.

To load and plot the eeg-alcohol dataset from the library:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # initialize the time series object
    ts = TimeSeries()

    # load and normalize the dataset from the library
    ts.load_series(utils.search_path("eeg-alcohol"), normalizer="z_score")

    # print and plot a subset of time series
    ts.print(nbr_series=4, nbr_val=4)
    ts.plot(input_data=ts.data, nbr_series=6, nbr_val=100, save_path="./imputegap_assets")



The module ``ts.datasets`` contains all the publicly available datasets provided by the library. They can be listed as follows:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    ts = TimeSeries()
    print(f"ImputeGAP datasets : {ts.datasets}")






To load your own dataset, add the path to your file in the ``ts.load_series`` function.


.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    ts = TimeSeries()
    ts.load_series("./my_path/my_file.txt")
    ts.print()


To import the time series as a matrix, pass it as an argument of the ``ts.import_matrix`` function:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    ts = TimeSeries()
    mat = [[2,3,9], [3,10,5], [-1,4,5], [0,0,0]]
    ts.import_matrix(mat)
    ts.print()




.. raw:: html

   <br>



.. _contamination:

GenGap: Missing Data Simulation
-------------------------------
ImputeGAP implements a module to simulate different missingness patterns in time series. The GenGap module can be invoked externally. The patterns are described `here <missingness_patterns.html>`_.

As example, we show how to contaminate the eeg-alcohol dataset with the MCAR pattern:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    from imputegap.recovery.contamination import GenGap
    from imputegap.tools import utils

    # initialize the time series object
    ts = TimeSeries()

    # load and normalize the dataset
    ts.load_series(utils.search_path("eeg-alcohol"), normalizer="z_score")

    # contaminate the time series with MCAR pattern
    ts_m = GenGap.mcar(ts.data, rate_dataset=0.2, rate_series=0.4, block_size=10, seed=True)

    # plot the contaminated time series
    ts.plot(ts.data, ts_m, nbr_series=9, subplot=True, save_path="./imputegap_assets/contamination")




All missingness patterns developed in ImputeGAP are available in the ``ts.patterns`` module. They can be listed as follows:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    ts = TimeSeries()
    print(f"Missingness patterns : {ts.patterns}")





.. raw:: html

   <br>




.. _imputation:

Imputation
----------

In this section, we will illustrate how to impute the contaminated time series. Our library implements six families of imputation algorithms: Statistical, Machine Learning, Matrix Completion, Deep Learning, Pattern Search, and Large Language Models.
The list of algorithms is described `here <algorithms.html>`_.


Let's illustrate the imputation using the CDRec algorithm from the Matrix Completion family.

.. code-block:: python

    from imputegap.recovery.imputation import Imputation
    from imputegap.recovery.contamination import GenGap
    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # initialize the time series object
    ts = TimeSeries()

    # load and normalize the dataset
    ts.load_series(utils.search_path("eeg-alcohol"), normalizer="z_score")

    # contaminate the time series
    ts_m = GenGap.mcar(ts.data)

    # impute the contaminated series
    imputer = Imputation.MatrixCompletion.CDRec(ts_m)
    imputer.impute()

    # compute and print the imputation metrics
    imputer.score(ts.data, imputer.recov_data)
    ts.print_results(imputer.metrics)

    # plot the recovered time series
    ts.plot(input_data=ts.data, incomp_data=ts_m, recov_data=imputer.recov_data, nbr_series=9, subplot=True, algorithm=imputer.algorithm, save_path="./imputegap_assets/imputation")


Imputation can be performed using either default values or user-defined values. To specify the parameters, please use a dictionary in the following format:

.. code-block:: python

    imputer.impute(params={"rank": 5, "epsilon": 0.01, "iterations": 100})


All algorithms developed in ImputeGAP are available in the ``ts.algorithms`` module. They can be listed as follows:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    ts = TimeSeries()
    print(f"Imputation families : {ts.families}")
    print(f"Imputation algorithms : {ts.algorithms}")



.. raw:: html

   <br>



.. _parameterization:

Parameter Tuning
----------------

The Optimizer component manages algorithm configuration and hyperparameter tuning. The parameters are defined by providing a dictionary containing the ground truth, the chosen optimizer, and the optimizer's options. Several search algorithms are available, including those provided by `Ray Tune <https://docs.ray.io/en/latest/tune/index.html>`_.

.. code-block:: python

    from imputegap.recovery.imputation import Imputation
    from imputegap.recovery.contamination import GenGap
    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # initialize the time series object
    ts = TimeSeries()

    # load and normalize the dataset
    ts.load_series(utils.search_path("eeg-alcohol"), normalizer="z_score")

    # contaminate and impute the time series
    ts_m = GenGap.mcar(ts.data)
    imputer = Imputation.MatrixCompletion.CDRec(ts_m)

    # use Ray Tune to fine tune the imputation algorithm
    imputer.impute(user_def=False, params={"input_data": ts.data, "optimizer": "ray_tune"})

    # compute the imputation metrics with optimized parameter values
    imputer.score(ts.data, imputer.recov_data)

    # compute the imputation metrics with default parameter values
    imputer_def = Imputation.MatrixCompletion.CDRec(ts_m).impute()
    imputer_def.score(ts.data, imputer_def.recov_data)

    # print the imputation metrics with default and optimized parameter values
    ts.print_results(imputer_def.metrics, text="Default values")
    ts.print_results(imputer.metrics, text="Optimized values")

    # plot the recovered time series
    ts.plot(input_data=ts.data, incomp_data=ts_m, recov_data=imputer.recov_data, nbr_series=9, subplot=True, algorithm=imputer.algorithm, save_path="./imputegap_assets/imputation")

    # save hyperparameters
    utils.save_optimization(optimal_params=imputer.parameters, algorithm=imputer.algorithm, dataset="eeg-alcohol", optimizer="ray_tune")




All optimizers developed in ImputeGAP are available in the ``ts.optimizers`` module. They can be listed as follows:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    ts = TimeSeries()
    print(f"AutoML Optimizers : {ts.optimizers}")



.. raw:: html

   <br>




.. _benchmark:

Benchmark
---------

ImputeGAP can serve as a common test-bed for comparing the effectiveness and efficiency of time series imputation algorithms [33]_.  Users have full control over the benchmark by customizing various parameters, including the list of the algorithms to compare, the optimizer, the datasets to evaluate, the missingness patterns, the range of missing values, and the performance metrics.


The benchmarking module can be utilized as follows:

.. code-block:: python

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

    # display the plot
    bench.subplots.show()





You can enable the optimizer using the following command:

.. code-block:: python

    opt = {"optimizer": "ray_tune", "options": {"n_calls": 1, "max_concurrent_trials": 1}}
    my_opt = [opt]


.. [33] Mourad Khayati, Alberto Lerner, Zakhar Tymchenko, Philippe Cudré-Mauroux: Mind the Gap: An Experimental Evaluation of Imputation of Missing Values Techniques in Time Series. Proc. VLDB Endow. 13(5): 768-782 (2020)


.. raw:: html

   <br>



.. _downstream:

Downstream
----------

ImputeGAP includes a dedicated module for systematically evaluating the impact of data imputation on downstream tasks. Currently, forecasting is the primary supported task, with plans to expand to additional tasks in the future.

.. code-block:: python

    from imputegap.recovery.imputation import Imputation
    from imputegap.recovery.contamination import GenGap
    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # initialize the time series object
    ts = TimeSeries()

    # load and normalize the timeseries
    ts.load_series(utils.search_path("forecast-economy"), normalizer="z_score")

    # contaminate the time series
    ts_m = GenGap.aligned(ts.data, rate_series=0.8)

    # define and impute the contaminated series
    imputer = Imputation.MatrixCompletion.CDRec(ts_m)
    imputer.impute()

    # compute and print the downstream results
    downstream_config = {"task": "forecast", "model": "hw-add", "baseline": "ZeroImpute"}
    imputer.score(ts.data, imputer.recov_data, downstream=downstream_config)
    ts.print_results(imputer.downstream_metrics, text="Downstream results")




All downstream models developed in ImputeGAP are available in the ``ts.forecasting_models`` module. They can be listed as follows:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    ts = TimeSeries()
    print(f"ImputeGAP downstream models for forecasting : {ts.forecasting_models}")






.. raw:: html

   <br>


.. _explainer:

Explainer
---------


The library provides insights into the algorithm's behavior by identifying the features that impact the imputation results. It trains a regression model to predict imputation results across various methods and uses SHapley Additive exPlanations (`SHAP <https://shap.readthedocs.io/en/latest/>`_) to measure how different time series features influence the model’s predictions.

Let's illustrate the explainer using the CDRec algorithm and MCAR missingness pattern:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    from imputegap.recovery.explainer import Explainer
    from imputegap.tools import utils

    # initialize the time series and explainer object
    ts = TimeSeries()
    exp = Explainer()

    # load and normalize the dataset
    ts.load_series(utils.search_path("eeg-alcohol"), normalizer="z_score")

    # configure the explanation
    exp.shap_explainer(input_data=ts.data, extractor="pycatch", pattern="mcar", file_name=ts.name, algorithm="CDRec")

    # print the impact of each feature
    exp.print(exp.shap_values, exp.shap_details)

    # plot the features impact
    exp.show()


All feature extractors developed in ImputeGAP are available in the ``ts.extractors`` module. They can be listed as follows:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    ts = TimeSeries()
    print(f"ImputeGAP features extractors : {ts.extractors}")


.. raw:: html

   <br>




