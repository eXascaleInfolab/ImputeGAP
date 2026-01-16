====================
Missingness Patterns
====================

ImputeGAP introduces a new taxonomy of missingness patterns tailored to time series, going beyond the traditional MAR and MNAR categories, which were not designed for temporal data.

.. _setup:

Setup
-----

.. note::

    -   M : number of time series
    -   N : length of time series
    -   W : user-defined offset window in the beginning of the series; default = 20
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

The missing blocks are aligned.

.. note::

    -   ``R ∈ [1%, (100-W)%]``
    -   The size of a single missing block varies between 1% and (100 - ``W``)% of ``N``.
    -   The starting position is the same and begins at ``W`` and progresses until the size of the missing block is reached, affecting the first series from the top up to ``S%`` of the dataset.


.. raw:: html

   <br>


.. image:: _img/aligned.png
   :alt: ImputeGAP Aligned Pattern
   :align: left
   :class: portrait

``ts_m = GenGap.aligned(ts.data, rate_dataset=0.6, rate_series=0.4, offset=25)``

.. raw:: html

   <br><br>


**Disjoint**

The missing blocks are disjoint.

.. note::

    -   ``R ∈ [1%, (100-W)%]``
    -   The size of a single missing block varies between 1% and (100 - ``W``)% of ``N``.
    -   The starting position of the first missing block begins at ``W``.
    -   Each subsequent missing block starts immediately after the previous one ends, continuing this pattern until the limit of the dataset or ``N`` is reached.


.. raw:: html

   <br>

.. image:: _img/disjoint.png
   :alt: ImputeGAP Disjoint Pattern
   :align: left
   :class: portrait

``ts_m = GenGap.disjoint(ts.data, rate_series=0.4, offset=25)``


.. raw:: html

   <br><br>


**Overlap**

The missing blocks are overlapping.

.. note::

    -   ``R ∈ [1%, (100-W)%]``
    -   The size of a single missing block varies between 1% and (100 - ``W``)% of ``N``.
    -   The starting position of the first missing block begins at ``W``.
    -   Each subsequent missing block starts after the previous one ends, but with a shift back of ``X%``, creating an overlap.
    -   This pattern continues until the limit or ``N`` is reached.

.. raw:: html

   <br>

.. image:: _img/overlap.png
   :alt: ImputeGAP Overlap Pattern
   :align: left
   :class: portrait

``ts_m = GenGap.overlap(ts.data, rate_series=0.4, offset=25)``


.. raw:: html

   <br><br>



**Scattered**

The missing blocks are scattered.

.. note::

    -   ``R ∈ [1%, (100-W)%]``
    -   The size of a single missing block varies between 1% and (100 - ``W``)% of ``N``.
    -   The starting position is randomly shifted by adding a random value to ``W``, then progresses until the size of the missing block is reached, affecting the first series from the top up to ``S%`` of the dataset.


.. raw:: html

   <br>

.. image:: _img/scatter.png
   :alt: ImputeGAP Scatter Pattern
   :align: left
   :class: portrait

``ts_m = GenGap.scattered(ts.data, rate_dataset=0.6, rate_series=0.4, offset=25)``

.. raw:: html

   <br><br>


.. _scenario_multi_block:

MULTI-BLOCK
-----------

Multiple missing blocks per series

.. raw:: html

   <br>

**MCAR**

The blocks are missing completely at random

.. note::

    -   ``R ∈ [1%, (100-W)%]``
    -   Data blocks of the same size are removed from arbitrary series at a random position between ``W`` and ``N``, until the total number of missing values per series is reached.


.. raw:: html

   <br>

.. image:: _img/mcar.png
   :alt: ImputeGAP Aligned Pattern
   :align: left
   :class: portrait

``ts_m = GenGap.mcar(ts.data, rate_dataset=0.6, rate_series=0.4, block_size=2, offset=25)``


.. raw:: html

   <br><br>


**Block Distribution**

The missing blocks follow a distribution.

.. note::

    -   ``R ∈ [1%, (100-W)%]``
    -   Data is removed following a distribution given by the user for every values of the series, affecting the first series from the top up to ``S%`` of the dataset.

To configure the block distribution pattern, please refer to this `page <tutorials_distribution.html>`_.


.. raw:: html

   <br>

.. image:: _img/distribution.png
   :alt: ImputeGAP Distribution Pattern
   :align: left
   :class: portrait


``ts_m = GenGap.distribution(ts.data, rate_dataset=0.6, rate_series=0.4, probabilities_list=probs, offset=25)``


.. raw:: html

   <br><br>