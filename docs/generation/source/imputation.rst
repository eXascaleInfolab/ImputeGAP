==========
Imputation
==========


In this section, we will illustrate how to impute the contaminated time series. Our library implements six families of imputation algorithms: Statistical, Machine Learning, Matrix Completion, Deep Learning, Pattern Search, and Large Language Models.
The list of algorithms is described here: `imputation algorithms <index.html#algorithms>`_.

.. _imputationtuto:

Imputation Tutorial
-------------------

Let's illustrate the imputation using the CDRec algorithm from the Matrix Completion family.

.. code-block:: python

    from imputegap.recovery.imputation import Imputation
    from imputegap.recovery.contamination import GenGAP
    from imputegap.recovery.manager import TimeSeries

    # initialize the time series object
    ts = TimeSeries()

    # load and normalize the dataset
    ts.load_series("eeg-alcohol", normalizer="z_score")

    # contaminate the time series
    ts_m = GenGAP.mcar(ts.data)

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

    imputer.impute(params={"rank": 5, "epsilon": 0.01, "iteration": 100})


For Deep Learning and LLM-based algorithms, the computing device (CPU or GPU) can be specified manually or selected automatically through the imputation parameters:

.. code-block:: python

    imputer = Imputation.LLMs.GPT4TS(ts_m, use_cuda=1)



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
    from imputegap.recovery.contamination import GenGAP
    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # initialize the time series object
    ts = TimeSeries()

    # load and normalize the dataset
    ts.load_series("eeg-alcohol", normalizer="z_score")

    # contaminate and impute the time series
    ts_m = GenGAP.mcar(ts.data)
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






.. raw:: html

   <br>

