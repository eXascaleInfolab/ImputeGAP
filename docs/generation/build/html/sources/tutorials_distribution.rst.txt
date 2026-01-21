============
Distribution
============

To use any distribution, provide a probability matrix as a parameter, covering all values and series. The following example illustrates the computation of a Gaussian distribution:

.. code-block:: python

    from scipy.stats import norm
    import numpy as np

    probabilities = []
    for series in ts.data:
        N = len(series)
        P = int(N * 0.1)
        R = np.arange(P, N)
        mean = np.mean(series)
        D = norm.pdf(R, loc=P + mean * (N - P), scale=0.2 * (N - P))
        D /= D.sum()
        probabilities.append(D)
    ts_m2 = ts.Contamination.distribution(ts.data, rate_dataset=0.2, rate_series=0.4, probabilities_list=probabilities, seed=True)



The code will produce the same output then this snippet :

.. code-block:: python

    ts_m = ts.Contamination.gaussian(ts.data, rate_dataset=0.2, rate_series=0.4, std_dev=0.2, seed=True)


