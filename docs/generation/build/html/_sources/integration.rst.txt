Integration
===========

ImputeGAP allows users to integrate their own algorithms. We describe in turn the integration python and other languages.

Initialization
~~~~~~~~~~~~~~

Initialize a Git Repository::

    git init
    git clone https://github.com/eXascaleInfolab/ImputeGAP
    cd ImputeGAP/
    source imputegap_source.sh


.. raw:: html

    <br><br>



Library Architecture
--------------------

ImputeGAP is organized into six main directories, each responsible for a specific aspect of the framework.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - **Directory**
     - **Description**
   * - ``algorithms/``
     - Contains the interfaces and execution logic used to invoke imputation algorithms, including integrations with external implementations.
   * - ``datasets/``
     - Stores the datasets provided with the library, along with additional datasets used for experimentation and evaluation.
   * - ``env/``
     - Defines default configurations, parameters, and settings for the supported algorithms and models.
   * - ``recovery/``
     - Contains the core modules responsible for time series imputation, recovery management, benchmarking, and evaluation.
   * - ``tools/``
     - Provides shared utilities and helper functions used throughout the library.
   * - ``wrapper/``
     - Contains external algorithm implementations, including code adapted from research papers and third-party sources.


.. note::

   To integrate an algorithm in C++, please follow the :doc:`integration tutorial <integration_c>`.


.. raw:: html

    <br><br>


A. Python Integration Steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Basic Features
--------------

1. Navigate to the ``./imputegap/algorithms`` directory.
2. Create a new file by copying ``mean_impute.py`` and rename it with the name of your algorithm, e.g., ``new_alg.py``.
3. Rename the function ``def mean_impute()``, e.g., ``def new_alg()``.
4. Replace the section under ``# core of the algorithm`` with your algorithm’s implementation. The algorithms should take as input the ``TimeSeries`` object structure and should return a ``numpy.ndarray`` matrix.
5. Navigate to ``./imputegap/recovery/imputation.py`` and search ``#mean-impute-class-contribution``:

   a. Copy the ``class MeanImpute(BaseImputer)`` into the corresponding class of algorithms' family.

   b. Rename the class. e.g., ``class NewAlg(BaseImputer)``.

   c. Change the value of the ``algorithm`` variable from ``mean_impute`` to ``new_alg``

   d. In the ``def impute()`` method, replace the call of the function to link into your new algorithm, e.g.,

    .. code-block:: python

        from imputegap.algorithms.new_alg import new_alg
        self.recov_data = new_alg(self.incomp_data, **params)


.. raw:: html

   <br><br>


Advanced Features
-----------------


I. Initialize default values
____________________________

1. To set the default values of your algorithm, please update ``./imputegap/env/default_values.toml`` (lines 3-6) and add your configuration. For example::

        [new_alg]
        param_integer = 42
        param_float = 0.42
        param_string = "value_42"


2. Update the ``./imputegap/tools/utils.py`` file, and search ``#contributing-load-params``, and specify your configuration in the ``load_parameters`` function.


.. raw:: html

   <br>


II. Benchmark and Toolkit
_________________________

To access the benchmarking features, please update ``./imputegap/tools/utils.py`` by adding your algorithm in the ``def config_impute_algorithm`` function (search ``#contributing-add-toolkit``) .

    .. code-block:: python


        elif algorithm == "your_algo_name":
            imputer = Imputation.MyFamily.NewAlg(incomp_data)


Replace MyFamily with either: Statistics, MatrixCompletion, PatternSearch, MachineLearning, DeepLearning, or LLMs.


.. raw:: html

   <br>


III. Optimizer
______________

To enable the optimization module, please update ``./imputegap/tools/algorithm_parameters.py``.

1. Open ``./imputegap/tools/algorithm_parameters.py``, search ``#contributing-optimization`` copy paste the lines and update the algorithm name and parameters, e.g.,

    .. code-block:: python

        'your_algo_name': {
                "param_integer": tune.grid_search([i for i in range(2, 20 1)]),
                "param_float": tune.loguniform(1e-6, 1),
                "param_string": ["value_1", "value_n"]
            },


2. Add your parameters in the ``def save_optimization()`` (search ``#contributing-optimization-save``) function of the file ``./imputegap/tools/utils.py`` to save the optimal parameters:

    .. code-block:: python

        if algorithm == "your_algo_name":
            params_to_save = {
                "param_integer": int(optimal_params[0]),
                "param_float": float(optimal_params[1]),
                "param_string": str(optimal_params[2])
        }


.. raw:: html

   <br>
