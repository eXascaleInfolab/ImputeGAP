==========
Downstream
==========


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


All downstream models implemented within ImputeGAP are available in the ``ts.forecasters`` module, which can be listed as follows:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    ts = TimeSeries()
    print(f"ImputeGAP downstream models for forecasting : {ts.forecasters}")



.. raw:: html

   <br>



To evaluate and compare the impact of multiple imputation algorithms on downstream forecasting performance, use the `eval()` method. The following example compares CDRec, MICE, and ZeroImpute using the Chronos forecasting model:

.. code-block:: python

    # ImputeGAP imports
    from imputegap.recovery.manager import TimeSeries
    from imputegap.recovery.contamination import GenGAP
    from imputegap.recovery.downstream import Downstream

    # initialize the time series object
    ts = TimeSeries()
    print(f"\nImputeGAP downstream models for forcasting : {ts.forecasters}")

    # load and normalize the dataset
    ts.load_series(data="chlorine", normalizer="z_score")

    # contaminate the time series
    ts_m = GenGAP.mcar(input_data=ts.data, rate_series=0.4, offset=0, reverse=True)

    # define the downstream options
    downstream = Downstream(input_data=ts.data, incomp_data=ts_m)

    # launch the analysis
    comparison = downstream.eval(imp_algs=["SAITS", "ZeroImpute"],
                                 model="chronos",
                                 horizons=[128],
                                 season=24,
                                 meta_pattern="mcar",
                                 meta_rate=0.4,
                                 use_cuda=2)


.. raw:: html

   <br>