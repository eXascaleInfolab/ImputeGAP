Integration
===========

ImputeGAP allows users to integrate their own algorithms. In this page, we describe how to integrate algorithms implemented in Python and other languages.

Initialization
--------------

Initialize the GitHub repository and install the requirements::

    git clone https://github.com/eXascaleInfolab/ImputeGAP
    cd ImputeGAP/
    source imputegap_source.sh



Basic Features
--------------

We will use the ``mean_impute`` algorithm as a template for integration. Throughout this guide, ``new_alg`` will serve as an example placeholder. Replace it with the name of your own algorithm.


.. rst-class:: integration-steps

1. Inside ``./imputegap/algorithms``, rename the file ``mean_impute.py`` to ``new_alg.py``.

2. Rename the function ``def mean_impute()`` to ``def new_alg()``, and adapt the parameters and their default values.

3. Replace the section under ``# core of the algorithm`` with your algorithm’s implementation. The algorithm must accept a contaminated matrix as input and return a recovered matrix as output, both in ``numpy.ndarray`` format.

4. Open ``./imputegap/recovery/imputation.py`` and search ``#mean-impute-class-contribution``:

   a. Copy the ``class MeanImpute(BaseImputer)`` code into the appropriate algorithm family and rename it ``class NewAlg(BaseImputer):``.

   b. Change the value of the ``algorithm`` variable from ``mean_impute`` to ``new_alg``.

   c. In the ``def impute()`` method, update the function call to invoke your new algorithm.

    .. code-block:: python

        from imputegap.algorithms.new_alg import new_alg
        self.recov_data = new_alg(incomp_data=self.incomp_data, logs=self.logs, verbose=self.verbose, use_cuda=self.use_cuda, **params)


5. To set the default values of your algorithm, please update ``./imputegap/env/default_values.toml``. Search ``#env-params-contribution``, uncomment the lines and add your configuration::

    [new_alg]
    param_integer = 42
    param_float = 0.42
    param_string = "value_42"


6. Update the ``./imputegap/tools/utils.py`` file, and search ``#contributing-load-params``, uncomment the lines, and specify your configuration to match the one in the env file.

.. code-block:: python

    elif algorithm == "new_alg":
        param_integer = int(config[algorithm]['param_integer'])
        param_float = float(config[algorithm]['param_float'])
        param_string = str(config[algorithm]['param_string'])
        return {"param_integer": param_integer,"param_float": param_float, "param_string": param_string}


7. To access the benchmarking utilities, please update ``./imputegap/tools/utils.py``. Search for ``#contributing-add-toolkit``, uncomment the relevant section, and add your algorithm.

.. code-block:: python

    elif alg == "new_alg":
        imputer = Imputation.YourFamily.NewAlg(incomp_data)



8. The integration is complete. You can now test it by running an imputation call:

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
    imputer = Imputation.YourFamily.NewAlg(incomp_data=ts_m, use_cuda=0)
    imputer.impute()

    # compute and print the imputation metrics
    imputer.score(input_data=ts.data, recov_data=imputer.recov_data)


.. raw:: html

   <br>


Optimizer
_________

To enable the optimization module, please update ``./imputegap/tools/algorithm_parameters.py``.

.. rst-class:: integration-steps

1. Search for ``#contributing-optimization``, uncomment the relevant lines, update the algorithm name and change parameter ranges.

    .. code-block:: python

        "new_alg": {
            "param_integer": tune.grid_search(list(range(2, 20, 1))),
            "param_float": tune.loguniform(1e-6, 1),
            "param_string": tune.choice(["value_1", "value_n"])
        },


2. Add your parameters in the ``def save_optimization()`` (search ``#contributing-optimization-save``) function of the file ``./imputegap/tools/utils.py`` to save the optimal parameters.

    .. code-block:: python

        elif algorithm == "new_alg":
            params_to_save = {
                "param_integer": int(optimal_params[0]),
                "param_float": float(optimal_params[1]),
                "param_string": str(optimal_params[2])
            }


.. raw:: html

   <br>


Contribution
------------

ImputeGAP endorses well-established algorithms for integration. The visibility conditions need to be met

    -   The paper must be published in a peer-review venue
    -   The results must be fully reproducible
    -   The algorithm handles time series imputation


To ensure reproducibility, the following technical requirements must be fulfilled:

    - The algorithm's output must preserve the matrix format used throughout the library.
    - All existing unit tests must continue to pass after integration.
    - A new unit test must be implemented using ``tests/test_imputation_cdrec.py`` as a template. It must reproduce the results reported in the original paper, with the expected values recorded in ``imputegap/env/imputegap_results.toml``.


.. raw:: html

    <br>


Library Folder Structure
------------------------

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

    <br>

