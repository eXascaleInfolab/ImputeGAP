=========
Tutorials
=========


.. _loading:

Dataset Loading
---------------

ImputeGAP comes with several time series datasets. The datasets are listed here: `data_loading.html <data_loading.html>`_.

As an example, we use the eeg-alcohol dataset, composed of individuals with a genetic predisposition to alcoholism. The dataset contains measurements from 64 electrodes placed on subject’s scalps, sampled at 256 Hz. The dimensions of the dataset are 64 series, each containing 256 values.

To load and plot the eeg-alcohol dataset from the library:

.. code-block:: python

    # ImputeGAP imports
    from imputegap.recovery.manager import TimeSeries

    # initialize the time series object
    ts = TimeSeries()
    print(f"\nImputeGAP datasets : {ts.datasets}")

    # load and normalize the dataset
    ts.load_series(data="eeg-alcohol", normalizer=None)

    # plot and print a subset of time series
    ts.print(nbr_series=4, nbr_val=4)
    ts.plot(input_data=ts.data, nbr_series=9, nbr_val=100, save_path="./imputegap_assets")



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

GenGAP: Missing Data Simulation
-------------------------------
GenGAP is a module to generate realistic missing gaps in sensor data. It allows users to fully customize all the properties of the missing gaps: start, size, position, frequency, etc., and can be easily imported inside any project.

As example, we show how to contaminate the eeg-alcohol dataset with the MCAR pattern:

.. code-block:: python

    # ImputeGAP imports
    from imputegap.recovery.manager import TimeSeries
    from imputegap.recovery.contamination import GenGAP

    # initialize the time series object
    ts = TimeSeries()
    print(f"\nMissingness patterns : {ts.patterns}")

    # load and normalize the dataset
    ts.load_series(data="eeg-alcohol", normalizer="z_score")

    # contaminate the time series with MCAR pattern
    ts_m = GenGAP.mcar(input_data=ts.data, rate_dataset=0.2, rate_series=0.4, block_size=10, seed=True)

    # plot the contaminated time series
    ts.plot(input_data=ts.data, incomp_data=ts_m, nbr_series=9, subplot=True, save_path="./imputegap_assets/contamination")




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
The list of algorithms is described here: `index.html#algorithms <index.html#algorithms>`_.


Let's illustrate the imputation using the CDRec algorithm from the Matrix Completion family.

.. code-block:: python

    # ImputeGAP imports
    from imputegap.recovery.imputation import Imputation
    from imputegap.recovery.contamination import GenGAP
    from imputegap.recovery.manager import TimeSeries

    # initialize the time series object
    ts = TimeSeries()
    print(f"\nImputation algorithms : {ts.algorithms}")

    # load and normalize the dataset
    ts.load_series(data="eeg-alcohol", normalizer="z_score", nbr_series=32, nbr_val=128)

    # contaminate the time series
    ts_m = GenGAP.mcar(input_data=ts.data)

    # impute the contaminated series
    imputer = Imputation.MatrixCompletion.CDRec(incomp_data=ts_m, use_cuda=0)
    imputer.impute()

    # compute and print the imputation metrics
    imputer.score(input_data=ts.data, recov_data=imputer.recov_data)

    # plot the recovered time series
    ts.plot(input_data=ts.data, incomp_data=ts_m, recov_data=imputer.recov_data, nbr_series=9, subplot=True, algorithm=imputer.algorithm, save_path="./imputegap_assets/imputation")


Imputation can be performed using either default values or user-defined values. To specify the parameters, please use a dictionary in the following format:

.. code-block:: python

    imputer.impute(params={"rank": 5, "epsilon": 0.01, "iteration": 100})


CPU or GPU can be selected manually or automatically through the imputation parameters. If the algorithm does not support GPU acceleration, CPU is used by default. Set `0` for CPU, `1` for GPU, or `2` for automatic device selection.

.. code-block:: python

    imputer = Imputation.LLMs.GPT4TS(incomp_data=ts_m, use_cuda=1)



All algorithms developed in ImputeGAP are available in the ``ts.algorithms`` module. They can be listed as follows:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    ts = TimeSeries()
    print(f"Imputation families : {ts.families}")
    print(f"Imputation algorithms : {ts.algorithms}")



To impute a dataset that already contains missing values (NaNs), load it directly and pass the marix to the imputation algorithm, bypassing the contamination step. Note that evaluation metrics and visualizations are unavailable without the complete dataset.

.. code-block:: python

    # ImputeGAP imports
    from imputegap.recovery.imputation import Imputation
    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # initialize the time series object
    ts = TimeSeries()
    print(f"\nImputation algorithms : {ts.algorithms}")

    # load the dataset that contains nans values
    ts.load_series(data="test-logic-nans-llm", normalizer=None)

    # skip the contamination and impute the series
    imputer = Imputation.MatrixCompletion.CDRec(incomp_data=ts.data, use_cuda=0)
    imputer.impute()
    print(f"Reconstructed data:\n{imputer.recov_data[:10]}")

    # save the imputed data
    path = utils.ts_caching_save(data=imputer.recov_data, type="imputation", dir_cache="./_caching")
    print(f"\nReconstruction saved: {path}")


.. raw:: html

   <br>



.. _parameterization:

Parameter Tuning
----------------

The Optimizer component manages algorithm configuration and hyperparameter tuning. The parameters are defined by providing a dictionary containing the ground truth, the chosen optimizer, and the optimizer's options. Several search algorithms are available, including those provided by `Ray Tune <https://docs.ray.io/en/latest/tune/index.html>`_.

.. code-block:: python

    # ImputeGAP imports
    from imputegap.recovery.imputation import Imputation
    from imputegap.recovery.contamination import GenGAP
    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # initialize the time series object
    ts = TimeSeries()

    # load and normalize the dataset
    ts.load_series(data="eeg-alcohol", normalizer="z_score")

    # contaminate and impute the time series
    ts_m = GenGAP.mcar(input_data=ts.data)
    imputer = Imputation.MatrixCompletion.CDRec(incomp_data=ts_m)
    imputer.verbose = False

    # use Ray Tune to fine tune the imputation algorithm
    imputer.impute(user_def=False, params={"input_data": ts.data, "optimizer": "ray_tune"})

    # compute the imputation metrics with default parameter values
    imputer_def = Imputation.MatrixCompletion.CDRec(incomp_data=ts_m).impute()
    imputer_def.score(input_data=ts.data, recov_data=imputer_def.recov_data, text="Default values")

    # compute the imputation metrics with optimized parameter values
    imputer.score(input_data=ts.data, recov_data=imputer.recov_data, text="Optimized values")

    # plot the recovered time series
    ts.plot(input_data=ts.data, incomp_data=ts_m, recov_data=imputer.recov_data, nbr_series=9, subplot=True, algorithm=imputer.algorithm, save_path="./imputegap_assets/imputation", display=True)

    # save hyperparameters
    utils.save_optimization(optimal_params=imputer.parameters, algorithm=imputer.algorithm, dataset="eeg-alcohol", optimizer="ray_tune")






.. raw:: html

   <br>




.. _benchmark:

BenchGAP
--------

BenchGAP is a module for benchmarking the performance (efficacy and runtime) of time series imputation algorithms. Users have full control over BenchGAP by customizing various parameters, including the list of the algorithms to compare, the optimizer, the datasets to evaluate, the missingness patterns, the range of missing values, and the performance metrics.

The benchmarking module can be utilized as follows:

.. code-block:: python

    # ImputeGAP import
    from imputegap.recovery.benchmark import BenchGAP

    # launch the evaluation
    bench = BenchGAP()
    bench.eval_imputation(algorithms=["MeanImpute", "CDRec"],
               datasets=["eeg-alcohol"],
               patterns=["mcar"],
               x_axis=[0.1, 0.2, 0.4, 0.6, 0.8],
               use_cache=False)

    # display the plot
    bench.subplots.show()



.. [33] Mourad Khayati, Alberto Lerner, Zakhar Tymchenko, Philippe Cudré-Mauroux: Mind the Gap: An Experimental Evaluation of Imputation of Missing Values Techniques in Time Series. Proc. VLDB Endow. 13(5): 768-782 (2020)


.. raw:: html

   <br>



.. _downstream:

Downstream
----------

ImputeGAP includes a dedicated module for systematically evaluating the impact of data imputation on downstream tasks. Currently, forecasting is the primary supported task, with plans to expand to additional tasks in the future.

.. code-block:: python

    # ImputeGAP imports
    from imputegap.recovery.imputation import Imputation
    from imputegap.recovery.contamination import GenGAP
    from imputegap.recovery.manager import TimeSeries
    from imputegap.recovery.downstream import Artifact, Forecaster

    # initialize the time series object
    ts = TimeSeries()
    print(f"\nImputeGAP downstream models for forcasting : {ts.forecasters}")

    # load and normalize the dataset
    ts.load_series(data="chlorine", normalizer="z_score")

    # contaminate the time series
    ts_m = GenGAP.mcar(input_data=ts.data, rate_series=0.2, offset=0, reverse=True)

    # define and impute the contaminated series
    imputer = Imputation.MatrixCompletion.CDRec(incomp_data=ts_m)
    imputer.impute()

    # define the parameters for the downstream model
    forecaster = Forecaster(artifact=Artifact(model="chronos", horizon=128, season=24), pipeline=True)

    # compute the downstream analysis and display the results
    imputer.score(input_data=ts.data, recov_data=imputer.recov_data, downstream=forecaster)




All downstream models developed in ImputeGAP are available in the ``ts.forecasters`` module. They can be listed as follows:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    ts = TimeSeries()
    print(f"ImputeGAP downstream models for forecasting : {ts.forecasters}")






.. raw:: html

   <br>


.. _explainer:

ExplainGAP
----------

ExplainGAP module provides insights into the algorithm's behavior by identifying the features that impact the imputation results. It trains a regression model to predict imputation results across various methods and uses SHapley Additive exPlanations (`SHAP <https://shap.readthedocs.io/en/latest/>`_) to reveal how different time series features influence the model’s predictions.

Let's illustrate the explainer using the CDRec algorithm and MCAR missingness pattern:

.. code-block:: python

    # ImputeGAP imports
    from imputegap.recovery.manager import TimeSeries
    from imputegap.recovery.explainer import ExplainGAP

    # initialize the time series and explainer object
    ts = TimeSeries()
    exp = ExplainGAP()
    print(f"\nImputeGAP explainer features extractor : {ts.extractors}")

    # load and normalize the dataset
    ts.load_series(data="eeg-alcohol", normalizer="z_score")

    # configure the explanation
    exp.shap_explainer(input_data=ts.data, extractor="pycatch", pattern="mcar", file_name=ts.name, algorithm="CDRec")

    # print the impact of each feature
    exp.print(shap_values=exp.shap_values, shap_details=exp.shap_details)

    # plot the feature impacts
    exp.show()


All feature extractors developed in ImputeGAP are available in the ``ts.extractors`` module. They can be listed as follows:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    ts = TimeSeries()
    print(f"ImputeGAP features extractors : {ts.extractors}")


.. raw:: html

   <br>




