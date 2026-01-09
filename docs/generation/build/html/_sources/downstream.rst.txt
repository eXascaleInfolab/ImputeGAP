==========
Downstream
==========

ImputeGAP includes a dedicated module for systematically evaluating the impact of data imputation on downstream tasks. Currently, forecasting is the primary supported task, with plans to expand to additional tasks in the future.

.. code-block:: python

    from imputegap.recovery.imputation import Imputation
    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # initialize the time series object
    ts = TimeSeries()

    # load and normalize the timeseries
    ts.load_series(utils.search_path("forecast-economy"))
    ts.normalize()

    # contaminate the time series
    ts_m = ts.Contamination.aligned(ts.data, rate_series=0.8)

    # define and impute the contaminated series
    imputer = Imputation.MatrixCompletion.CDRec(ts_m)
    imputer.impute()

    # compute and print the downstream results
    downstream_config = {"task": "forecast", "model": "hw-add", "comparator": "ZeroImpute"}
    imputer.score(ts.data, imputer.recov_data, downstream=downstream_config)
    ts.print_results(imputer.downstream_metrics, algorithm=imputer.algorithm)




All downstream models developed in ImputeGAP are available in the ``ts.forecasting_models`` module, which can be listed as follows:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    ts = TimeSeries()
    print(f"ImputeGAP downstream models for forecasting : {ts.forecasting_models}")






.. raw:: html

   <br>