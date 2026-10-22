:orphan:

Integration C++
===============

ImputeGAP allows users to integrate their own algorithms. We describe in turn the integration python and other languages.

Initialization
~~~~~~~~~~~~~~

Initialize a Git Repository::

    git init
    git clone https://github.com/eXascaleInfolab/ImputeGAP
    cd ImputeGAP/
    source imputegap_source.sh



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




B. C++ Integration Steps
~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   To integrate an algorithm in Python, please follow the :doc:`integration tutorial <integration>`.

.. raw:: html

    <br><br>




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
