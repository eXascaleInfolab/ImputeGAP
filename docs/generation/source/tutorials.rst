=========
Tutorials
=========


.. _loading-preprocessing:

Loading and Preprocessing
-------------------------

ImputeGAP comes with several time series datasets. The list of datasets is described `here <datasets.html>`_.

As an example, we start by using eeg-alcohol, a standard dataset composed of individuals with a genetic predisposition to alcoholism. The dataset contains measurements from 64 electrodes placed on subject’s scalps, sampled at 256 Hz (3.9-ms epoch) for 1 second. The dimensions of the dataset are 64 series, each containing 256 values.

The library enables time series normalization as a preprocessing step prior to imputation. Users can select from two normalization techniques to standardize their data distribution.

    - Z-score normalization: Standardizes data by subtracting the mean and dividing by the standard deviation, ensuring a mean of 0 and a standard deviation of 1.
    - Min-max normalization: Scales data to a fixed range, typically [0,1], by adjusting values based on the minimum and maximum in the dataset.

You can access the API documentation at the following `link <imputegap.manager.html#imputegap.recovery.manager.TimeSeries.normalize>`_.


.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # initialize the time series object
    ts = TimeSeries()

    # load and normalize the dataset from file or from the code
    ts.load_series(utils.search_path("eeg-alcohol"))
    ts.normalize(normalizer="z_score")

    # plot and print a subset of time series
    ts.plot(input_data=ts.data, nbr_series=9, nbr_val=100, save_path="./imputegap_assets")
    ts.print(nbr_series=9, nbr_val=20)



The module ``ts.datasets`` contains all the publicly available datasets provided by the library.

To list all the datasets, you can use this command:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    ts = TimeSeries()
    print(f"ImputeGAP datasets : {ts.datasets}")






.. raw:: html

   <br><br>



.. _contamination:

Contamination
-------------
We now describe how to simulate missing values in the loaded dataset. ImputeGAP implements eight different missingness patterns. The list of patterns is described `here <patterns.html>`_.

As example, we show how to contaminate the eeg-alcohol dataset with the MCAR pattern:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # initialize the time series object
    ts = TimeSeries()

    # load and normalize the dataset
    ts.load_series(utils.search_path("eeg-alcohol"))
    ts.normalize(normalizer="z_score")

    # contaminate the time series with MCAR pattern
    ts_m = ts.Contamination.mcar(ts.data, rate_dataset=0.2, rate_series=0.4, block_size=10, seed=True)

    # plot the contaminated time series
    ts.plot(ts.data, ts_m, nbr_series=9, subplot=True, save_path="./imputegap_assets/contamination")




All missingness patterns developed in ImputeGAP are available in the ``ts.patterns`` module.

To list all the available patterns, you can use this command:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    ts = TimeSeries()
    print(f"Missingness patterns : {ts.patterns}")





.. raw:: html

   <br><br>




.. _imputation:

Imputation
----------

In this section, we will illustrate how to impute the contaminated time series. Our library implements five families of imputation algorithms. Statistical, Machine Learning, Matrix Completion, Deep Learning, and Pattern Search Methods.
The list of algorithms is described `here <algorithms.html>`_.


Let's illustrate the imputation using the CDRec Algorithm from the Matrix Completion family.

.. code-block:: python

    from imputegap.recovery.imputation import Imputation
    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # initialize the time series object
    ts = TimeSeries()

    # load and normalize the dataset
    ts.load_series(utils.search_path("eeg-alcohol"))
    ts.normalize(normalizer="z_score")

    # contaminate the time series
    ts_m = ts.Contamination.mcar(ts.data)

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

    config = {"rank": 5, "epsilon": 0.01, "iterations": 100}
    imputer.impute(params=config)


All algorithms developed in ImputeGAP are available in the ``ts.algorithms`` module.

To list all the available algorithms, you can use this command:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    ts = TimeSeries()
    print(f"Imputation algorithms : {ts.algorithms}")



.. raw:: html

   <br><br>



.. _parameterization:

Parameter Tuning
----------------

The Optimizer component manages algorithm configuration and hyperparameter tuning. To invoke the tuning process, users need to specify the optimization option during the Impute call by selecting the appropriate input for the algorithm. The parameters are defined by providing a dictionary containing the ground truth, the chosen optimizer, and the optimizer's options. Several search algorithms are available, including those provided by `Ray Tune <https://docs.ray.io/en/latest/tune/index.html>`_.

.. code-block:: python

    from imputegap.recovery.imputation import Imputation
    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # initialize the time series object
    ts = TimeSeries()

    # load and normalize the dataset
    ts.load_series(utils.search_path("eeg-alcohol"))
    ts.normalize(normalizer="z_score")

    # contaminate and impute the time series
    ts_m = ts.Contamination.mcar(ts.data)
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
    ts.plot(input_data=ts.data, incomp_data=ts_m, recov_data=imputer.recov_data, nbr_series=9, subplot=True, save_path="./imputegap_assets/imputation", display=True)

    # save hyperparameters
    utils.save_optimization(optimal_params=imputer.parameters, algorithm=imputer.algorithm, dataset="eeg-alcohol", optimizer="ray_tune", file_name="./imputegap_assets/params"




All optimizers developed in ImputeGAP are available in the ``ts.optimizers`` module.

To list all the available optimizers, you can use this command:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    ts = TimeSeries()
    print(f"AutoML Optimizers : {ts.optimizers}")



.. raw:: html

   <br><br>




.. _benchmark:

Benchmark
---------

ImputeGAP can serve as a common test-bed for comparing the effectiveness and efficiency of time series imputation algorithms [33]_.  Users have full control over the benchmark by customizing various parameters, including the list of datasets to evaluate, the algorithms to compare, the choice of optimizer to fine-tune the algorithms on the chosen datasets, the missingness patterns, and the range of missing rates. The default metrics evaluated include "RMSE", "MAE", "MI", "Pearson", and the runtime.


The benchmarking module can be utilized as follows:

.. code-block:: python

    from imputegap.recovery.benchmark import Benchmark

    save_dir = "./imputegap_assets/benchmark"
    nbr_runs = 1

    datasets = ["eeg-alcohol"]

    optimizers = ["default_params"]

    algorithms = ["SoftImpute", "KNNImpute"]

    patterns = ["mcar"]

    range = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

    # launch the evaluation
    list_results, sum_scores = Benchmark().eval(algorithms=algorithms, datasets=datasets, patterns=patterns, x_axis=range, optimizers=optimizers, save_dir=save_dir, runs=nbr_runs)





You can change the optimizer using the following command:

.. code-block:: python

    optimizer = {"optimizer": "ray_tune", "options": {"n_calls": 1, "max_concurrent_trials": 1}}
    optimizers = [optimizer]


.. [33] Mourad Khayati, Alberto Lerner, Zakhar Tymchenko, Philippe Cudré-Mauroux: Mind the Gap: An Experimental Evaluation of Imputation of Missing Values Techniques in Time Series. Proc. VLDB Endow. 13(5): 768-782 (2020)


.. raw:: html

   <br><br>



.. _downstream:

Downstream
----------


ImputeGAP includes a dedicated module for systematically evaluating the impact of data imputation on downstream tasks. Currently, forecasting is the primary supported task, with plans to expand to additional tasks in the future.

Below is an example of how to call the downstream process for the model by defining a dictionary with the task and the name the model:

.. code-block:: python

    from imputegap.recovery.imputation import Imputation
    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # initialize the time series object
    ts = TimeSeries()

    # load and normalize the timeseries
    ts.load_series(utils.search_path("forecast-economy"))
    ts.normalize(normalizer="min_max")

    # contaminate the time series
    ts_m = ts.Contamination.aligned(ts.data, rate_series=0.8)

    # define and impute the contaminated series
    imputer = Imputation.MatrixCompletion.CDRec(ts_m)
    imputer.impute()

    # compute and print the downstream results
    downstream_config = {"task": "forecast", "model": "hw-add", "comparator": "ZeroImpute"}
    imputer.score(ts.data, imputer.recov_data, downstream=downstream_config)
    ts.print_results(imputer.downstream_metrics, algorithm=imputer.algorithm)




To list all the available downstream models, you can use this command:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    ts = TimeSeries()
    print(f"ImputeGAP downstream models for forcasting : {ts.downstream_models}")






.. raw:: html

   <br><br>


.. _explainer:

Explainer
---------


ImputeGAP provides insights into the algorithm's behavior by identifying the features that impact the most the imputation results. It trains a regression model to predict imputation results across various methods and uses SHapley Additive exPlanations (`SHAP <https://shap.readthedocs.io/en/latest/>`_) to reveal how different time series features influence the model’s predictions.

Let's illustrate the explainer using the cdrec Algorithm and MCAR missingness pattern:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    from imputegap.recovery.explainer import Explainer
    from imputegap.tools import utils

    # initialize the time series object
    ts = TimeSeries()

    # load and normalize the timeseries
    ts.load_series(utils.search_path("eeg-alcohol"))
    ts.normalize(normalizer="z_score")

    # configure the explanation
    shap_values, shap_details = Explainer.shap_explainer(input_data=ts.data, extractor="pycatch", pattern="mcar", file_name=ts.name, algorithm="CDRec")

    # print the impact of each feature
    Explainer.print(shap_values, shap_details)


To list all the available features extractors, you can use this command:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    ts = TimeSeries()
    print(f"ImputeGAP features extractors : {ts.extractors}")


.. raw:: html

   <br><br>




