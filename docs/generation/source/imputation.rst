==========
Imputation
==========


In this section, we will illustrate how to impute the contaminated time series. Our library implements six families of imputation algorithms: Statistical, Machine Learning, Matrix Completion, Deep Learning, Pattern Search, and Large Language Models.
The list of algorithms is described here: `imputation algorithms <index.html#algorithms>`_.

.. _imputationtuto:

Imputation of simulated missing data
------------------------------------

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

    imputer = Imputation.LLMs.GPT4TS(ts_m, use_cuda=1)



All algorithms developed in ImputeGAP are available in the ``ts.algorithms`` module. They can be listed as follows:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    ts = TimeSeries()
    print(f"Imputation families : {ts.families}")
    print(f"Imputation algorithms : {ts.algorithms}")



.. raw:: html

   <br>



.. _imputationincomplete:

Imputation of raw missing data
------------------------------

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

