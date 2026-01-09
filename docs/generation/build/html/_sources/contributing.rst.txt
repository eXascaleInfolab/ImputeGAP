Contributing
============

ImputeGAP allows users to integrate their own algorithms. We describe in turn the integration python and other languages.

Initialization
~~~~~~~~~~~~~~

Initialize a Git Repository::

        $ git init
        $ git clone https://github.com/eXascaleInfolab/ImputeGAP
        $ cd ./ImputeGAP


A. Python Integration Steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Basic Features
--------------

1. Navigate to the ``./imputegap/algorithms`` directory.
2. Create a new file by copying ``mean_impute.py`` and rename it with the name of your algorithm, e.g., ``new_alg.py``.
3. Rename the function ``def mean_impute()``, e.g., ``def new_alg()``.
4. Replace the section under ``# core of the algorithm`` with your algorithm’s implementation. The algorithms should take as input the ``TimeSeries`` object structure and should return a ``numpy.ndarray`` matrix.
5. Navigate to ``./imputegap/recovery/imputation.py``:

   a. Copy the ``class MeanImpute(BaseImputer)`` into the corresponding class of algorithms' family.

   b. Rename the class. e.g., ``class NewAlg(BaseImputer)``.

   c. Change the value of the ``algorithm`` variable from ``mean_impute`` to ``new_alg``

   d. In the ``def impute()`` method, replace the call of the function to link into your new algorithm, e.g.,

    .. code-block:: python

        from imputegap.algorithms.new_alg import new_alg
        self.recov_data = new_alg(self.incomp_data, params)


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


2. Update the ``./imputegap/tools/utils.py`` file, and specify your configuration in the ``load_parameters`` function.


.. raw:: html

   <br>


II. Benchmark
_____________

To access the benchmarking features, please update ``./imputegap/tools/utils.py`` (lines 547-550) by adding your algorithm in the ``def config_impute_algorithm`` function.

    .. code-block:: python


        elif algorithm == "new_alg":
            imputer = Imputation.MyFamily.NewAlg(incomp_data)


Replace MyFamily with either: Statistics, MatrixCompletion, PatternSearch, MachineLearning, DeepLearning, or LLMs.


.. raw:: html

   <br>


III. Optimizer
______________

To enable the optimization module, please update ``./imputegap/tools/algorithm_parameters.py``.

1. Open ``./imputegap/tools/algorithm_parameters.py`` copy paste lines 296 to 302 and update the algorithm name and parameters, e.g.,

    .. code-block:: python

        'new_alg': {
                "param_integer": tune.grid_search([i for i in range(2, 20 1)]),
                "param_float": tune.loguniform(1e-6, 1),
                "param_string": ["value_1", "value_n"]
            },


2. Add your parameters in the ``def save_optimization()`` function of the file ``./imputegap/tools/utils.py`` to save the optimal parameters, line 808 to 815:

    .. code-block:: python

        if algorithm == "new_alg":
            params_to_save = {
                "param_integer": int(optimal_params[0]),
                "param_float": float(optimal_params[1]),
                "param_string": str(optimal_params[2])
        }


.. raw:: html

   <br>



IV. Update the call
___________________

Navigate to ``./imputegap/recovery/imputation.py``:

Improve the imputation call of the ``NewAlg`` class in the ``def impute()`` function, and add the call of the optimizer and the default values of the parameters.

.. code-block:: python

    if params is not None:
        param_integer, param_float, param_string = self._check_params(user_def, params)  # call the optimizer
    else:
        param_integer, param_float, param_string = utils.load_parameters(query="default", algorithm=self.algorithm, verbose=self.verbose)  # load the default values

    self.recov_data = new_alg(incomp_data=self.incomp_data, param_integer=param_integer, param_float=param_float, param_string=param_string, logs=self.logs, verbose=self.verbose)



.. raw:: html

   <br><br>


B. C++ Integration Steps
~~~~~~~~~~~~~~~~~~~~~~~~
We provide a wrapper that can serve as a template for the integration of users’ code. We will show how to adjust the wrapper in C++.

1. Navigate to the ``./imputegap/algorithms`` directory.
2. Convert your CPP/H files into a shared object format (``.so``) and place them in the ``imputegap/algorithms/lib`` folder.

   a. Go to ``./imputegap/wrapper/AlgoCollection`` and update the Makefile. Copy commands from ``libSTMVL.so`` or modify them as needed.

   b. Optionally, copy your C++ project files into the directory.

   c. Generate the ``.so`` file using the ``make`` command::

        make your_lib_name

   d. To include the .so file in the "in-built" directory, open a command line, navigate to the root directory, and execute the library build process::

        rm -rf dist/
        python setup.py sdist bdist_wheel

3. Rename ``cpp_integration.py`` to the name of your algorithm.
4. Modify the ``native_algo()`` function:

   a. Update the shared object parameter to match your shared library.

   b. Convert input parameters to the appropriate C++ types and pass them to your shared object methods.

   c. Convert the imputed matrix back to a numpy format.

5. Adapt the template method ``your_algo.py`` with the appropriate parameters, ensuring compatibility with the ``TimeSeries`` object and a ``numpy.ndarray`` return type.
6. Adapt the ``./imputegap/recovery/imputation.py`` by adding a function to call your new algorithm by copying and modifying ``class MeanImpute(BaseImputer)`` as needed. You can copy-paste the class into the corresponding category of algorithms.

7. Perform imputation as needed.

.. raw:: html

   <br><br>

Example with C++ Algorithm
--------------------------

Once your cpp and h files are ready to be converted (you can look at ``./imputegap/wrapper/AlgoCollection/shared/SharedLibCDREC.cpp`` or ``./imputegap/wrapper/AlgoCollection/shared/SharedLibCDREC.h``), create a ``.so`` file for linux and windows, and a ``.dylib`` file for MAC OS.

.. tabs::
    .. tab:: Windows

        1. Modify the Makefile::

            libCDREC.so:
                g++ -O3 -D ARMA_DONT_USE_WRAPPER -fPIC -rdynamic -shared -o lib_cdrec.so -Wall -Werror -Wextra -pedantic \
                -Wconversion -Wsign-conversion -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -fopenmp -std=gnu++14 \
                Stats/Correlation.cpp Algorithms/CDMissingValueRecovery.cpp  Algebra/Auxiliary.cpp \
                Algebra/CentroidDecomposition.cpp  shared/SharedLibCDREC.cpp \
                -lopenblas -larpack

        2. Generate the shared library::

            make libCDREC.so

        3. Place the generated ``.so`` file in ``imputegap/algorithms/lib``
        4. Optional: To include the .so file in the "in-built" directory::

            rm -rf dist/
            python setup.py sdist bdist_wheel

        .. raw:: html

            <br><br>

    .. tab:: Linux

        1. Modify the Makefile::

            libCDREC.so:
                g++ -O3 -D ARMA_DONT_USE_WRAPPER -fPIC -rdynamic -shared -o lib_cdrec.so -Wall -Werror -Wextra -pedantic \
                -Wconversion -Wsign-conversion -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -fopenmp -std=gnu++14 \
                Stats/Correlation.cpp Algorithms/CDMissingValueRecovery.cpp  Algebra/Auxiliary.cpp \
                Algebra/CentroidDecomposition.cpp  shared/SharedLibCDREC.cpp \
                -lopenblas -larpack

        2. Generate the shared library::

            make libCDREC.so

        3. Place the generated ``.so`` file in ``imputegap/algorithms/lib``
        4. Optional: To include the .so file in the "in-built" directory::

            rm -rf dist/
            python setup.py sdist bdist_wheel

        .. raw:: html

            <br><br>

    .. tab:: MacOs

        1. Modify the Makefile::

            libCDREC.dylib:
                clang++ -dynamiclib -O3 -fPIC -std=c++17 -o lib_cdrec.dylib \
                -I/opt/homebrew/include \
                -L/opt/homebrew/lib \
                -L/opt/homebrew/opt/openblas/lib \
                Stats/Correlation.cpp Algorithms/CDMissingValueRecovery.cpp Algebra/Auxiliary.cpp \
                Algebra/CentroidDecomposition.cpp shared/SharedLibCDREC.cpp \
                -larmadillo -lopenblas -larpack
        2. Generate the shared library::

            make libCDREC.dylib

        3. Place the generated ``.dylib`` file in ``imputegap/algorithms/lib``
        4. Optional: To include the .dylib file in the "in-built" directory::

            rm -rf dist/
            python setup.py sdist bdist_wheel

        .. raw:: html

           <br><br>

.. raw:: html

   <br><br>

**Wrapper**


1. In ``imputegap/algorithms/cpp_integration.py``, update the function name and parameter count, and ensure the ``.so`` file matches::

    def native_cdrec(__py_matrix, __py_rank, __py_epsilon, __py_iterations):

        shared_lib = utils.load_share_lib("lib_cdrec") # in-build files
        # shared_lib = utils.load_share_lib("./your_path/lib_cdrec.so") # external files

2. Convert variables to corresponding C++ types::

        __py_n = len(__py_matrix);
        __py_m = len(__py_matrix[0]);

        assert (__py_rank >= 0);
        assert (__py_rank < __py_m);
        assert (__py_epsilon > 0);
        assert (__py_iterations > 0);

        __ctype_size_n = __native_c_types_import.c_ulonglong(__py_n);
        __ctype_size_m = __native_c_types_import.c_ulonglong(__py_m);

        __ctype_rank = __native_c_types_import.c_ulonglong(__py_rank);
        __ctype_epsilon = __native_c_types_import.c_double(__py_epsilon);
        __ctype_iterations = __native_c_types_import.c_ulonglong(__py_iterations);

        __ctype_matrix = __marshal_as_native_column(__py_matrix);

3. Call the C++ algorithm with the required parameters::

        shared_lib.cdrec_imputation_parametrized(__ctype_matrix, __ctype_size_n, __ctype_size_m, __ctype_rank, __ctype_epsilon, __ctype_iterations);

4. Convert the imputed matrix back to ``numpy``::

        __py_imputed_matrix = __marshal_as_numpy_column(__ctype_matrix, __py_n, __py_m);

        return __py_imputed_matrix;

.. raw:: html

   <br><br>

**Method Implementation**

1. In ``imputegap/algorithms/cpp_integration.py``, create or adapt a generic method for your needs::

    def cdrec(contamination, truncation_rank, iterations, epsilon, logs=True, lib_path=None):

        start_time = time.time()  # Record start time

        # Call the C++ function to perform recovery
        imputed_matrix = native_cdrec(contamination, truncation_rank, epsilon, iterations)

        end_time = time.time()

        if logs:
            print(f"\n\t\t> logs, imputation cdrec - Execution Time: {(end_time - start_time):.4f} seconds\n")

        return imputed_matrix

.. raw:: html

   <br><br>

**Imputer Class**

1. Add your algorithm to the catalog in ``./imputegap/recovery/imputation.py``
2. Copy and modify ``class MeanImpute(BaseImputer)`` to fit your requirements::

    class MatrixCompletion:
        class CDRec(BaseImputer):
            algorithm = "cdrec"

            def impute(self, user_defined=True, params=None):

                self.imputed_matrix = cdrec(contamination=self.infected_matrix, truncation_rank=rank, iterations=iterations, epsilon=epsilon, logs=self.logs)

                return self
