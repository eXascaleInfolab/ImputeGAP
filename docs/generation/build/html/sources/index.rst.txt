ImputeGAP Documentation
=======================

.. admonition:: 📢 News !

        - [21-01-2026] Version 1.1.2 is out! The new release includes the following major updates:
            - New Deep Learning and LLMs algorithms
            - New utilities to handle multivariate datasets and algorithms
            - New module to simulate missing values: GenGap
        - [04-06-2025] ImputeGAP will be presented as a Tutorial at KDD 2025 in Toronto! `[Link] <https://imputegap-tutorials.github.io/KDD-2025/>`_


ImputeGAP is a comprehensive Python library for imputation of missing values in  time series data. It implements user-friendly APIs to easily visualize, analyze, and repair incomplete time series datasets. The library supports a diverse range of imputation algorithms and modular missing data simulation catering to datasets with varying characteristics. ImputeGAP includes extensive customization options, such as automated hyperparameter tuning, benchmarking, explainability, and downstream evaluation.


In detail, the library provides:
    - Over 40 state-of-the-art time series imputation algorithms from six different families
    - Several imputation univariate time series datasets and utilities to handle multivariate ones
    - Configurable contamination module that simulates real-world missingness patterns
    - AutoML techniques to parameterize the imputation algorithms
    - Unified benchmarking pipeline to evaluate the performance of imputation algorithms
    - Modular analysis tools to assess the impact of imputation on time series downstream tasks
    - Expainability module to understand the impact of time series features on the imputation results
    - Adjustable wrappers to integrate new algorithms in different languages: Python, C++, Matlab, Java, and R


.. raw:: html

   <br>



.. _get_started:

Getting Started
---------------

.. raw:: html

   <script>
      function applyTheme(e)
      {
        document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
      }

      const darkQuery = window.matchMedia('(prefers-color-scheme: dark)');
      applyTheme(darkQuery); // Apply on load
      darkQuery.addEventListener('change', applyTheme); // React to changes
   </script>

   <style>
        [data-theme="dark"] .card
        {
            padding: 15px;
            border-radius: 8px;
            background-color: #181818;
        }

        [data-theme="dark"] .card p
        {
            color: #CCCCCC;
        }

        [data-theme="dark"] .card h3 a
        {
            color: #2e86c1;
            text-decoration: none;
        }

        [data-theme="light"] .card
        {
            padding: 15px;
            border-radius: 8px;
            background-color: #f8f9fa;
        }

        [data-theme="light"] .card p
        {
            color: #333333;
        }

        [data-theme="light"] .card h3 a
        {
            color: #2e86c1;
            text-decoration: none;
        }
   </style>


   <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">

      <div class="card">
        <h3><a href="getting_started.html">🚀 Installation</a></h3>
        <p>Read the guide on how to install ImputeGAP on your system.</p>
      </div>

      <div class="card">
        <h3><a href="tutorials.html">📖 Tutorials</a></h3>
        <p>Check the tutorials to learn how to use ImputeGAP.</p>
      </div>

      <div class="card">
        <h3><a href="imputegap.html">📦 API</a></h3>
        <p>Find the main API for each submodule in the index.</p>
      </div>

      <div class="card">
        <h3><a href="algorithms.html">🧠 Algorithms</a></h3>
        <p>Explore the core algorithms used in ImputeGAP.</p>
      </div>

    </div><br>


.. raw:: html

   <br>



.. _cite:

Cite
----

If you like our library, please add a ⭐ in our `GitHub repository <https://github.com/eXascaleInfolab/ImputeGAP/>`_.

If you use ImputeGAP in your research, please cite the following papers:

.. code-block:: bash

    @article{nater2025imputegap,
      title = {ImputeGAP: A Comprehensive Library for Time Series Imputation},
      author = {Nater, Quentin and Khayati, Mourad and Pasquier, Jacques},
      year = {2025},
      eprint = {2503.15250},
      archiveprefix = {arXiv},
      primaryclass = {cs.LG},
      url = {https://arxiv.org/abs/2503.15250}


    @article{nater2025kdd,
      title = {A Hands-on Tutorial on Time Series Imputation with ImputeGAP},
      author = {Nater, Quentin and Khayati, Mourad and Cudré-Mauroux, Philippe},
      year = {2025},
      booktitle = {SIGKDD Conference on Knowledge Discovery and Data Mining (To Appear)},
      series = {KDD2025}
    }



.. raw:: html

   <br>



.. toctree::
   :maxdepth: 0
   :caption: Contents:
   :hidden:


   index
   getting_started
   tutorials
   datasets
   algorithms
   missingness_patterns
   notebooks
   contributing
   about_us
   imputegap


