=============
Preprocessing
=============

ImputeGAP enables time series normalization as a preprocessing step prior to imputation. Users can select from two normalization techniques to standardize their data distribution.

    - Z-score normalization: Standardizes data by subtracting the mean and dividing by the standard deviation, ensuring a mean of 0 and a standard deviation of 1.
    - Min-max normalization: Scales data to a fixed range, typically [0,1], by adjusting values based on the minimum and maximum in the dataset.

You can access the API documentation at the following `link <imputegap.manager.html#imputegap.recovery.manager.TimeSeries.normalize>`_.

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries
    from imputegap.tools import utils

    # initialize the time series object
    ts = TimeSeries()

    # load the timeseries from file or from the code
    ts.load_series(utils.search_path("eeg-alcohol"), nbr_series=9, nbr_val=100)
    ts.normalize(normalizer="z_score")



