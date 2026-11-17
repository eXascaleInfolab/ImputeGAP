============
Data Loading
============

The library currently hosts 14 datasets spanning 8 domains. Those datasets were collected from various time series imputation works in recent years.


.. raw:: html

   <div class="gengap-contents">
      <h2>Contents</h2>

      <div class="gengap-nav">
         <a href="#loadingtutorial">📖 Loading Tutorial</a>
         <a href="#datasets">⚙️ Datasets</a>
      </div>
   </div>


.. _loadingtutorial:

Loading Tutorial
----------------

ImputeGAP comes with several time series datasets. As an example, we use the eeg-alcohol dataset, composed of individuals with a genetic predisposition to alcoholism. The dataset contains measurements from 64 electrodes placed on subject’s scalps, sampled at 256 Hz. The dimensions of the dataset are 64 series, each containing 256 values.

To load and plot the eeg-alcohol dataset from the library:

.. code-block:: python

    from imputegap.recovery.manager import TimeSeries

    # initialize the time series object
    ts = TimeSeries()

    # load and normalize the dataset from the library
    ts.load_series("eeg-alcohol", normalizer="z_score")

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



.. _datasets:

Datasets
--------

The datasets are stored in ``./imputegap/datasets`` and are represented using a ``numpy.ndarray`` structure.

The datasets follow this format:

- Columns are the series/features, separator: empty space

- Rows are the timestamps' values, separator: newline

- Missing values are marked as NaNs




.. list-table::
   :header-rows: 1
   :widths: 30 30
   :class: benchmark-table

   * - **Datasets**
     - **Categories**
   * - `climate <https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#Climate-Change-Dataset>`_
     - Weather
   * - `meteo <https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#MeteoSwiss-Weather-Dataset>`_
     - Weather
   * - `temperature <https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#China-Climate-Temperature-Dataset>`_
     - Weather
   * - `airq <https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#Air-Quality-Dataset>`_
     - Air Quality
   * - `drift <https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#Gas-Sensor-Array-Drift-Dataset>`_
     - Air Quality
   * - `bafu <https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#BAFU-Water-Discharge-Dataset>`_
     - Water Quality
   * - `chlorine <https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#Chlorine-Residual-Dataset>`_
     - Water Quality
   * - `eeg-alcohol <https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#EEG-Alcohol-Dataset>`_
     - Neuroscience
   * - `motion <https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#Motion-Sensor-Dataset>`_
     - IoT Sensing
   * - `traffic <https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#PEMS-Traffic-Dataset>`_
     - IoT Sensing
   * - `soccer <https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#Soccer-Player-Tracking-Dataset>`_
     - Sports
   * - `sport-activity <https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#Sport-Activity-Dataset>`_
     - Sports
   * - `electricity <https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#Electricity-Consumption-Dataset>`_
     - Energy
   * - `stock-exchange <https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#Stock-Exchange-Rates-Dataset>`_
     - Finance



.. raw:: html

   <br>


