====================
Missingness Patterns
====================

ImputeGAP introduces a new taxonomy of missingness patterns tailored to time series, going beyond the traditional MAR and MNAR categories, which were not designed for temporal data.

.. _setup:

Setup
-----

.. note::

    -   (N, M) : number of timestamps, number of series
    -   W : user-defined offset window in the beginning of the series; default = 25
    -   R : user-defined rate of missing values (%); default = 20%
    -   S : user-defined rate of contaminated series (%); default = 20%

.. raw:: html

   <br>


.. _scenario_mono_block:

MONO-BLOCK
----------
One missing block per series

.. raw:: html

   <br>

**Aligned**

Missing blocks start at the same selected positions and have the same fixed size across the chosen series, resulting in aligned missing intervals.

.. note::

    -   ``R ∈ [1%, (100-W)%]``
    -   The size of a single missing block varies between 1% and (100 - ``W``)% of ``N``.
    -   The starting position is the same and begins at ``W`` and progresses until the size of the missing block is reached, affecting the first series from the top up to ``S%`` of the dataset.
    -   ``GenGap.aligned(ts.data, rate_dataset=1, rate_series=0.4, offset=25)``


.. raw:: html

   <br>


.. image:: _img/imputegap_aligned.png
   :alt: ImputeGAP Aligned Pattern
   :align: left
   :class: portrait



.. raw:: html

   <br><br>


**Disjoint**

Each missing block begins where the previous one ends, so the missing intervals are consecutive and do not overlap.

.. note::

    -   ``R ∈ [1%, (100-W)%]``
    -   The size of a single missing block varies between 1% and (100 - ``W``)% of ``N``.
    -   The starting position of the first missing block begins at ``W``.
    -   Each subsequent missing block starts immediately after the previous one ends, continuing this pattern until the limit of the dataset or ``N`` is reached.
    -   ``GenGap.disjoint(ts.data, rate_series=0.4, offset=25)``


.. raw:: html

   <br>

.. image:: _img/imputegap_disjoint.png
   :alt: ImputeGAP Disjoint Pattern
   :align: left
   :class: portrait


.. raw:: html

   <br><br>


**Overlap**

Each missing block begins a specified gap after the end of the previous one, making the missing intervals consecutive and overlapping.

.. note::

    -   ``R ∈ [1%, (100-W)%]``
    -   The size of a single missing block varies between 1% and (100 - ``W``)% of ``N``.
    -   The starting position of the first missing block begins at ``W``.
    -   Each subsequent missing block starts after the previous one ends, but with a shift back of ``X%``, creating an overlap.
    -   This pattern continues until the limit or ``N`` is reached.
    -   ``GenGap.overlap(ts.data, rate_series=0.4, offset=25, shift=0.1)``

.. raw:: html

   <br>

.. image:: _img/imputegap_overlap.png
   :alt: ImputeGAP Overlap Pattern
   :align: left
   :class: portrait


.. raw:: html

   <br><br>



**Scattered**

The starting position of the missing block is chosen at random, all missing blocks share the same fixed size, with only one block removed per series.

.. note::

    -   ``R ∈ [1%, (100-W)%]``
    -   The size of a single missing block varies between 1% and (100 - ``W``)% of ``N``.
    -   The starting position is randomly shifted by adding a random value to ``W``, then progresses until the size of the missing block is reached, affecting the first series from the top up to ``S%`` of the dataset.
    -   ``GenGap.scattered(ts.data, rate_dataset=1, rate_series=0.4, offset=25)``


.. raw:: html

   <br>

.. image:: _img/imputegap_scattered.png
   :alt: ImputeGAP Scatter Pattern
   :align: left
   :class: portrait



.. raw:: html

   <br><br>


.. _scenario_multi_block:

MULTI-BLOCK
-----------

Multiple missing blocks per series

.. raw:: html

   <br>

**MCAR**

Missing blocks are introduced completely at random. Blocks are removed from randomly chosen positions, every removed block has the same fixed size, and the affected time series are selected at random.

.. note::

    -   ``R ∈ [1%, (100-W)%]``
    -   Data blocks of the same size are removed from arbitrary series at a random position between ``W`` and ``N``, until the total number of missing values per series is reached.
    -   ``GenGap.mcar(ts.data, rate_dataset=1, rate_series=0.2, offset=25, seed=False, block_size=20)``


.. raw:: html

   <br>

.. image:: _img/imputegap_mcar.png
   :alt: ImputeGAP Aligned Pattern
   :align: left
   :class: portrait



.. raw:: html

   <br><br>


**Block Distribution**

Missingness follows a probability distribution, each position has a certain chance of being missing.

.. note::

    -   ``R ∈ [1%, (100-W)%]``
    -   Data is removed following a distribution given by the user for every values of the series, affecting the first series from the top up to ``S%`` of the dataset.
    -   ``GenGap.gaussian(ts.data, rate_dataset=1, rate_series=0.4, offset=25, selected_mean="position", std_dev=0.2)``

To configure the block distribution pattern, please refer to this `page <tutorials_distribution.html>`_.


.. raw:: html

   <br>

.. image:: _img/imputegap_distribution.png
   :alt: ImputeGAP Distribution Pattern
   :align: left
   :class: portrait



.. raw:: html

   <br><br>