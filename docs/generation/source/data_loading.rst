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




.. _datasets:

Datasets
--------

The datasets are stored in ``./imputegap/datasets`` and are represented using a ``numpy.ndarray`` structure.

The datasets follow this format:

- Columns are the series/features, separator: empty space

- Rows are the timestamps' values, separator: newline

- Missing values are marked as NaNs




.. raw:: html

   <table class="benchmark-table docutils align-default">
     <thead>
       <tr><th>Categories</th><th>Datasets</th></tr>
     </thead>
     <tbody>
       <tr>
         <td rowspan="3">Weather</td>
         <td><a href="https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#Climate-Change-Dataset">climate</a></td>
       </tr>
       <tr><td><a href="https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#MeteoSwiss-Weather-Dataset">meteo</a></td></tr>
       <tr><td><a href="https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#China-Climate-Temperature-Dataset">temperature</a></td></tr>

       <tr>
         <td rowspan="2">Air Quality</td>
         <td><a href="https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#Air-Quality-Dataset">airq</a></td>
       </tr>
       <tr><td><a href="https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#Gas-Sensor-Array-Drift-Dataset">drift</a></td></tr>

       <tr>
         <td rowspan="2">Water Quality</td>
         <td><a href="https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#BAFU-Water-Discharge-Dataset">bafu</a></td>
       </tr>
       <tr><td><a href="https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#Chlorine-Residual-Dataset">chlorine</a></td></tr>

       <tr>
         <td>Neuroscience</td>
         <td><a href="https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#EEG-Alcohol-Dataset">eeg-alcohol</a></td>
       </tr>

       <tr>
         <td rowspan="2">IoT Sensing</td>
         <td><a href="https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#Motion-Sensor-Dataset">motion</a></td>
       </tr>
       <tr><td><a href="https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#PEMS-Traffic-Dataset">traffic</a></td></tr>

       <tr>
         <td rowspan="2">Sports</td>
         <td><a href="https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#Soccer-Player-Tracking-Dataset">soccer</a></td>
       </tr>
       <tr><td><a href="https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#Sport-Activity-Dataset">sport-activity</a></td></tr>

       <tr>
         <td>Energy</td>
         <td><a href="https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#Electricity-Consumption-Dataset">electricity</a></td>
       </tr>
       <tr>
         <td>Finance</td>
         <td><a href="https://github.com/eXascaleInfolab/ImputeGAP/tree/main/imputegap/datasets#Stock-Exchange-Rates-Dataset">stock-exchange</a></td>
       </tr>
     </tbody>
   </table>

.. raw:: html

   <br>


