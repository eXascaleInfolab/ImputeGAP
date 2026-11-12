============
Benchmarking
============

ImputeGAP can serve as a common test-bed for comparing the effectiveness and efficiency of time series imputation algorithms [33]_.  Users have full control over the benchmark by customizing various parameters, including the list of the algorithms to compare, the optimizer, the datasets to evaluate, the missingness patterns, the range of missing values, and the performance metrics.

.. raw:: html

   <p style="text-align: center; font-weight: bold;">
      Time Series → Contamination → Imputation → Evaluation
   </p><br />

.. raw:: html

   <div class="gengap-contents">

      <div class="gengap-nav">
         <a href="index.html#algorithms">▦ Algorithms</a>
         <a href="data_loading.html#datasets">▦ Datasets</a>
         <a href="#benchgaptuto">📖 BenchGAP Tutorial</a>
      </div>
   </div>



.. .. _imputationalgorithms:

.. Imputation Algorithms
.. ---------------------

.. .. list-table::
   .. :header-rows: 1
   :widths: 25 75
   :class: benchmark-table
    * - **Family**
    - **Algorithms**
    * - **Matrix Completion**
     - ``CDRec``, ``IterativeSVD``, ``GROUSE``, ``ROSL``, ``SPIRIT``, ``SoftImpute``, ``SVT``, ``TRMF``
    * - **Pattern Search**
     - ``STMVL``, ``DynaMMo``, ``TKCM``
    * - **Machine Learning**
     - ``IIM``, ``XGBOOST``, ``MICE``, ``MissForest``
    * - **Statistical**
     - ``KNNImpute``, ``Interpolation``, ``MinImpute``, ``MeanImpute``, ``ZeroImpute``, ``MeanImputeBySeries``
    * - **Deep Learning**
     - ``MRNN``, ``BRITS``, ``DeepMVI``, ``MPIN``, ``PRISTI``, ``MissNet``, ``GAIN``, ``GRIN``, ``BayOTIDE``, ``HKMFT``, ``BitGraph``, ``SAITS``, ``CSDI``, ``TimesNet``
    * - **LLMs**
     - ``NuwaTS``, ``GPT4TS``, ``Moment``


.. Missingness Patterns
.. --------------------

.. .. list-table::
   :header-rows: 1
   :widths: 33 33 34
   :class: benchmark-table
   * - ``blks``
     - ``seqn``
     - ``mcar``
   * - Block size
     - Sequence Number
     - Missing Completely at Random


.. Datasets
.. --------

.. .. list-table::
   :header-rows: 1
   :widths: 25 75
   :class: benchmark-table
   * - **Category**
     - **Datasets**
   * - **Weather**
     - ``climate``, ``meteo``, ``temperature``
   * - **Air Quality**
     - ``airq``, ``drift``
   * - **Water Quality**
     - ``bafu``, ``chlorine``
   * - **Neuroscience**
     - ``eeg-alcohol``
   * - **IoT Sensing**
     - ``motion``, ``traffic``
   * - **Sports**
     - ``soccer``, ``sport-activity``
   * - **Energy**
     - ``electricity``
   * - **Finance**
     - ``stock-exchange``


.. _benchgaptuto:

Benchmarking Tutorial
---------------------

The benchmarking module can be utilized as follows:

.. code-block:: python

    from imputegap.recovery.benchmark import BenchGAP

    # launch the evaluation
    bench = BenchGAP()
    bench.eval_imputation(algorithms=["MeanImpute", "SoftImpute"],
               datasets=["eeg-alcohol"],
               patterns=["mcar"],
               x_axis=[0.1, 0.2, 0.4, 0.6, 0.8])

    # display the plot
    bench.subplots.show()



.. [33] Mourad Khayati, Alberto Lerner, Zakhar Tymchenko, Philippe Cudré-Mauroux: Mind the Gap: An Experimental Evaluation of Imputation of Missing Values Techniques in Time Series. Proc. VLDB Endow. 13(5): 768-782 (2020)


.. raw:: html

   <br>



Output Directory
----------------

Each benchmark execution creates a unique experiment directory. The
directory tree below directly describes the purpose of each generated
folder and file:

.. code-block:: text

   CleanImp/
   │
   ├── _caching/                                  # Cache used to avoid recomputing expensive pipelines
   │   └── ...                                    # Imputed matrices and downstream predictions
   │
   ├── imputegap_assets/
   │   └── benchmark/
   │       └── <experiment>/                      # Unique directory created for the benchmark run
   │           │
   │           ├── <dataset>/                     # Results grouped by evaluated dataset
   │           │   └── <pattern>/                 # Results grouped by missingness pattern
   │           │       └── error/                 # Upstream/downstream metric results
   │           │           ├── _metrics_subplot.jpg
   │           │           │                      # Metric evolution across missingness rates
   │           │           └── report_<pattern>_<dataset>.txt
   │           │                                  # Detailed results for this dataset/pattern
   │           │
   │           ├── _heatmaps/                     # Aggregated results used for heatmap analyses
   │           │   └── _benchmarking_*.txt        # Benchmark results represented as heatmaps
   │           │
   │           ├── _summary_cleanimp_*.xlsx       # Complete summary of metrics and evaluated pipelines
   │           ├── experimentation_setup.txt      # Exact configuration of the benchmark run
   │           ├── report_cleanimp_benchmark_*.log # Complete benchmark results in log format
   │           └── runtime.log                    # Runtime information for the experiment
   │
   └── *_log.txt                                  # General execution logs


In short, the outputs are organized at three levels:

.. code-block:: text

   <experiment>
       ├── <dataset>/<pattern>/   → Detailed results
       ├── _heatmaps/             → Aggregated visual analysis
       └── summary/log files      → Global experiment results and configuration


This structure keeps every benchmark run self-contained while allowing
detailed results to be traced back to the corresponding dataset,
missingness pattern, and experiment configuration.



.. raw:: html

   <br>
