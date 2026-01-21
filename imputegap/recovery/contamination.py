import math
import numpy as np
from imputegap.tools import utils

class GenGap:
    """
    Class for contaminating times series data. This class is used to simulate missing values in the loaded dataset.

    Methods
    -------
    mcar(ts, series_rate=0.2, missing_rate=0.2, block_size=10, offset=0.1, seed=True, logic_by_series=True, explainer=False, verbose=True) :
            Apply Missing Completely at Random (MCAR) contamination to selected series.

    def aligned(input_data, rate_dataset=0.2, rate_series=0.2, offset=0.1, single_series=-1, logic_by_series=True, explainer=False, verbose=True):
        Apply missing percentage contamination to selected series.

    blackout(ts, missing_rate=0.2, offset=0.1, logic_by_series=True, verbose=True) :
        Apply blackout contamination to selected series.

    gaussian(input_data, series_rate=0.2, missing_rate=0.2, std_dev=0.2, offset=0.1, seed=True, logic_by_series=True, verbose=True):
        Apply Gaussian contamination to selected series.

    distribution(input_data, rate_dataset=0.2, rate_series=0.2, probabilities=None, offset=0.1, seed=True, logic_by_series=True, verbose=True):
        Apply any distribution contamination to the time series data based on their probabilities.

    disjoint(input_data, missing_rate=0.1, limit=1, offset=0.1, logic_by_series=True, verbose=True):
        Apply Disjoint contamination to selected series.

    overlap(input_data, missing_rate=0.2, limit=1, shift=0.05, offset=0.1, logic_by_series=True, verbose=True):
        Apply Overlapping contamination to selected series.

    References
    ----------
        https://imputegap.readthedocs.io/en/latest/patterns.html

    """

    def __init__(self, verbose=True):
        """
        Initialize the GenGAP object.
        """
        if verbose:
            print(f"ImputeGAP’s contamination module, GenGap, has been invoked (https://github.com/eXascaleInfolab/ImputeGAP).")


    def _compute_offset(N, offset):
        if offset < 1:
            return math.ceil(N * offset)  # values to protect in the beginning of the series
        else:
            return offset


    def mcar(input_data, rate_dataset=0.2, rate_series=0.2, block_size=10, offset=0.1, seed=True, logic_by_series=True, explainer=False, verbose=True):
        """
        Missing blocks are introduced completely at random. Time series are selected at random, and blocks of a fixed size are removed at randomly chosen positions.

        Docs: https://imputegap.readthedocs.io/en/latest/missingness_patterns.html


        Parameters
        ----------
        input_data : numpy.ndarray
            The time series dataset to contaminate.

        rate_dataset : float, optional
            Percentage of series to contaminate (default is 0.2).

        rate_series : float, optional
            Percentage of missing values per series (default is 0.2).

        block_size : int, optional
            Size of the block of missing data (default is 10).

        offset : float, optional
            Length of the initial uncontaminated segment of the series (default 0.1).
            If offset < 1, it is interpreted as a fraction of the total series length.
            If offset >= 1, it is interpreted as the exact number of initial values to keep uncontaminated.

        seed : bool, optional
            Whether to use a seed for reproducibility (default is True).

        logic_by_series : bool, optional
            Contaminate the series based on the series (sensor) malfunction (default: True).

        explainer : bool, optional
            Only used within the Explainer Module to contaminate one series at a time (default: False).

        verbose : bool, optional
            Whether to display the contamination information (default is True).

        Returns
        -------
        numpy.ndarray
            The contaminated time series data.

        Example
        -------
            >>> ts_m = GenGap.mcar(ts.data, rate_dataset=0.2, rate_series=0.4, block_size=10):

        """

        if logic_by_series:
            input_data = input_data.T # series-based contamination

        if seed:
            seed_value = 42
            if explainer:
                seed_value = 42+(int(rate_dataset)+1)
            #np.random.default_rng(seed_value)
            np.random.seed(seed_value)
        else:
            seed_value = -1

        ts_contaminated = input_data.copy()
        M, NS = ts_contaminated.shape

        if not explainer:  # use random series
            rate_series = utils.verification_limitation(rate_series)
            rate_dataset = utils.verification_limitation(rate_dataset)

            nbr_series_impacted = int(np.ceil(M * rate_dataset))
            series_selected = [str(idx) for idx in np.random.choice(M, nbr_series_impacted, replace=False)]

        else:  # use fix series
            series_selected = [str(rate_dataset)]

        if offset < 1:
            offset_nbr = math.ceil(offset * NS)
            if not explainer:
                offset = utils.verification_limitation(offset, low_limit=0)
        else:
            offset_nbr = offset

        values_nbr = int(NS * rate_series)

        if not explainer and verbose:
            print(f"\n(CONT) missigness pattern: MCAR"
                  f"\n\tselected series: {', '.join(str(int(n)+1) for n in sorted(series_selected, key=int))}"
                  f"\n\tpercentage of contaminated series: {rate_dataset * 100}%"
                  f"\n\trate of missing data per series: {rate_series * 100}%"
                  f"\n\tblock size: {block_size}"
                  f"\n\tsecurity offset: [0-{offset_nbr}]"
                  f"\n\tseed value: {seed_value}\n")


        if offset_nbr + values_nbr > NS:
            raise ValueError(
                f"\n\tError: The sum of offset ({offset_nbr}) and missing values ({values_nbr}) exceeds the limit of of the series."
                f" ({offset_nbr+values_nbr} must be smaller than {NS}).")


        # BLOCK CHECK
        S = int(series_selected[0])
        N = len(ts_contaminated[S])  # number of values in the series
        P = GenGap._compute_offset(N=N, offset=offset)
        W = int(N * rate_series)  # number of data to remove
        B = int(W / block_size)  # number of block to remove
        if B <= 0:
            print(f"\n\t(CORRECTION) The block size {block_size} is not be appropriate for this dataset shape {input_data.shape}.\n\tOne series has {N} values, with the offset, {N - P} values are available to contamination. Thus, the number of data to remove is {W} (int({N-P} * {rate_series})), but block size is {block_size} -> ({block_size} must be < {W})")
            block_size = W//2
            if block_size == 0:
                block_size = 1
            print(f"\t\t(ACTION) block_size is set to : {block_size}\n")


        for series in series_selected:
            S = int(series)
            N = len(ts_contaminated[S])  # number of values in the series
            P = GenGap._compute_offset(N=N, offset=offset)  # values to protect in the beginning of the series
            W = int(N * rate_series)  # number of data to remove
            B = int(W / block_size)  # number of block to remove

            if B <= 0:
                raise ValueError("The number of block to remove must be greater than 0. The dataset or the number of blocks may not be appropriate. One series has", str(N), "population is ", str((N - P)), "the number to remove str(W), and block site", str(block_size), "")

            data_to_remove = np.random.choice(range(P, N), B, replace=False)

            if np.isnan(ts_contaminated[S]).any():
                series_data = ts_contaminated[S]
                allowed_slice = series_data[P:]
                nans = np.isnan(allowed_slice).sum()
                removable = len(series_data[P:]) - nans
                required = B * block_size  # points we want to remove
                if removable <= 0 or removable < required:
                    print(f"[skip] series {S}: not enough points to remove. N={N}, removable={removable}, nans={nans}, required={required}")
                    continue

            for start_point in data_to_remove:
                for jump in range(block_size):  # remove the block size for each random position
                    position = start_point + jump

                    if position >= N:  # If block exceeds the series length
                        position = P + (position - N)  # Wrap around to the start after protection

                    while np.isnan(ts_contaminated[S, position]):
                        position = position + 1

                        if position >= N:  # If block exceeds the series length
                            position = P + (position - N)  # Wrap around to the start after protection

                    ts_contaminated[S, position] = np.nan

        if logic_by_series:
            return ts_contaminated.T
        else:
            return ts_contaminated


    def aligned(input_data, rate_dataset=0.2, rate_series=0.2, offset=0.1, single_series=-1, logic_by_series=True, explainer=False, verbose=True):
        """
        Missing blocks start and end at the same selected positions across the chosen series, resulting in aligned missing intervals.

        Docs: https://imputegap.readthedocs.io/en/latest/missingness_patterns.html


        Parameters
        ----------
        input_data : numpy.ndarray
            The time series dataset to contaminate.

        rate_dataset : float, optional
            Percentage of series to contaminate (default is 0.2).

        rate_series : float, optional
            Percentage of missing values per series (default is 0.2).

        offset : float, optional
            Length of the initial uncontaminated segment of the series (default 0.1).
            If offset < 1, it is interpreted as a fraction of the total series length.
            If offset >= 1, it is interpreted as the exact number of initial values to keep uncontaminated.

        single_series: int, optional
            Target only 1 series on the dataset depending on the ID provided (default is -1, which means, not set).

        logic_by_series : bool, optional
            Contaminate the series based on the series (sensor) malfunction (default: True).

        explainer : bool, optional
            Only used within the Explainer Module to contaminate one series at a time (default: False).

        verbose : bool, optional
            Whether to display the contamination information (default is True).

        Returns
        -------
        numpy.ndarray
            The contaminated time series data.

        Example
        -------
            >>> ts_m = GenGap.aligned(ts.data, rate_dataset=0.2, rate_series=0.4, offset=0.1):

        """

        if logic_by_series:
            input_data = input_data.T # series-based contamination

        ts_contaminated = input_data.copy()
        M, NS = ts_contaminated.shape
        default_init = 0

        if offset < 1: # percentage or real value
            offset_nbr = math.ceil(offset * NS)
            if not explainer:
                offset = utils.verification_limitation(offset, low_limit=0)
        else:
            offset_nbr = offset

        values_nbr = int(NS * rate_series)

        if not explainer:  # use random series
            rate_series = utils.verification_limitation(rate_series)
            rate_dataset = utils.verification_limitation(rate_dataset)
            nbr_series_impacted = int(np.ceil(M * rate_dataset))
        else:  # use fix series
            nbr_series_impacted = int(rate_dataset)
            default_init = nbr_series_impacted
            nbr_series_impacted = nbr_series_impacted + 1

        if single_series != -1:
            if single_series >= M:
                single_series = M-1
            default_init = single_series
            nbr_series_impacted = default_init+1
            rate_dataset = round(1/M)

        if not explainer and verbose:
            print(f"\n(CONT) missigness pattern: ALIGNED"
                  f"\n\tpercentage of contaminated series: {rate_dataset * 100}%"
                  f"\n\trate of missing data per series: {rate_series * 100}%"
                  f"\n\tsecurity offset: [0-{offset_nbr}]"
                  f"\n\ttimestamps impacted : {offset_nbr} -> {offset_nbr + values_nbr - 1}"
                  f"\n\tseries impacted : {default_init} -> {nbr_series_impacted-1}\n")

        if offset_nbr + values_nbr > NS:
            raise ValueError(f"\n\tError: The sum of offset ({offset_nbr}) and missing values ({values_nbr}) exceeds the limit of of the series.({offset_nbr+values_nbr} must be smaller than {NS}).")

        for series in range(default_init, nbr_series_impacted):
            S = int(series)
            N = len(ts_contaminated[S])  # number of values in the series
            P = GenGap._compute_offset(N=N, offset=offset)  # values to protect in the beginning of the series
            W = int(N * rate_series)  # number of data to remove

            for to_remove in range(0, W):
                index = P + to_remove
                ts_contaminated[S, index] = np.nan

        if logic_by_series:
            return ts_contaminated.T
        else:
            return ts_contaminated

    def scattered(input_data, rate_dataset=0.2, rate_series=0.2, offset=0.1, seed=True, logic_by_series=True, explainer=False, verbose=True):
        """
        The missing blocks all have the same size, but their starting positions are chosen at random.

        Docs: https://imputegap.readthedocs.io/en/latest/missingness_patterns.html

        Parameters
        ----------
        input_data : numpy.ndarray
            The time series dataset to contaminate.

        rate_dataset : float, optional
            Percentage of series to contaminate (default is 0.2).

        rate_series : float, optional
            Percentage of missing values per series (default is 0.2).

        offset : float, optional
            Size of the uncontaminated section at the beginning of the series (default is 0.1).

        seed : bool, optional
            Whether to use a seed for reproducibility (default is True).

        logic_by_series : bool, optional
            Contaminate the series based on the series (sensor) malfunction (default: True).

        explainer : bool, optional
            Only used within the Explainer Module to contaminate one series at a time (default: False).

        verbose : bool, optional
            Whether to display the contamination information (default is True).

        Returns
        -------
        numpy.ndarray
            The contaminated time series data.

        Example
        -------
            >>> ts_m = GenGap.scattered(ts.data, rate_dataset=0.2, rate_series=0.4, offset=0.1)

        """

        if logic_by_series:
            input_data = input_data.T # series-based contamination

        if seed:
            seed_value = 42
            np.random.default_rng(seed_value)
            #np.random.seed(seed_value)

        ts_contaminated = input_data.copy()
        M, NS = ts_contaminated.shape
        default_init = 0

        if offset < 1:  # percentage or real value
            offset_nbr = math.ceil(offset * NS)
            if not explainer:
                offset = utils.verification_limitation(offset, low_limit=0)
        else:
            offset_nbr = offset

        values_nbr = int(NS * rate_series)

        if not explainer:  # use random series
            rate_series = utils.verification_limitation(rate_series)
            rate_dataset = utils.verification_limitation(rate_dataset)
            nbr_series_impacted = int(np.ceil(M * rate_dataset))
        else:  # use fix series
            nbr_series_impacted = int(rate_dataset)
            default_init = nbr_series_impacted
            nbr_series_impacted = nbr_series_impacted + 1

        if not explainer and verbose:
            print(f"\n(CONT) missigness pattern: SCATTER"
                  f"\n\tpercentage of contaminated series: {rate_dataset * 100}%"
                  f"\n\trate of missing data per series: {rate_series * 100}%"
                  f"\n\tsecurity offset: [0-{offset_nbr}]"
                  f"\n\tindex impacted : {offset_nbr} -> {offset_nbr + values_nbr}\n")


        if offset_nbr + values_nbr > NS:
            raise ValueError(f"\n\tError: The sum of offset ({offset_nbr}) and missing values ({values_nbr}) exceeds the limit of of the series."
                f" ({offset_nbr+values_nbr} must be smaller than {NS}).")


        for series in range(default_init, nbr_series_impacted):
            S = int(series)
            N = len(ts_contaminated[S])  # number of values in the series
            P = GenGap._compute_offset(N=N, offset=offset)  # values to protect in the beginning of the series
            W = int(N * rate_series)  # number of data to remove
            L = (N - W - P) +1

            start_index = np.random.randint(0, L)  # Random start position

            for to_remove in range(0, W):
                index = P + start_index + to_remove
                ts_contaminated[S, index] = np.nan

        if logic_by_series:
            return ts_contaminated.T
        else:
            return ts_contaminated


    def blackout(input_data, rate_series=0.2, offset=0.1, logic_by_series=True, verbose=True):
        """
        Apply blackout contamination to selected series

        Parameters
        ----------
        input_data : numpy.ndarray
            The time series dataset to contaminate.

        rate_series : float, optional
            Percentage of missing values per series (default is 0.2).

        offset : float, optional
            Size of the uncontaminated section at the beginning of the series (default is 0.1).

        logic_by_series : bool, optional
            Contaminate the series based on the series (sensor) malfunction (default: True).

        verbose : bool, optional
            Whether to display the contamination information (default is True).

        Returns
        -------
        numpy.ndarray
            The contaminated time series data.

        Example
        -------
            >>> ts_m =GenGap.blackout(ts.data, series_rate=0.2)

        """
        return GenGap.aligned(input_data, rate_dataset=1, rate_series=rate_series, offset=offset, logic_by_series=logic_by_series, verbose=verbose)



    def gaussian(input_data, rate_dataset=0.2, rate_series=0.2, selected_mean="position", std_dev=0.2, offset=0.1, seed=True, logic_by_series=True, explainer=False, verbose=True):
        """
        Missingness follows a probability distribution, each position has a certain chance of being missing.

        Docs: https://imputegap.readthedocs.io/en/latest/missingness_patterns.html


        Parameters
        ----------
        input_data : numpy.ndarray
            The time series dataset to contaminate.

        rate_dataset : float, optional
            Percentage of series to contaminate (default is 0.2).

        rate_series : float, optional
            Percentage of missing values per series (default is 0.2).

        selected_mean: str, optional
            Strategy to compute the mean value (default : "position").
            Possibilities : "position", "values".

        std_dev : float, optional
            Standard deviation of the Gaussian distribution for missing values (default is 0.4).

        offset : float, optional
            Size of the uncontaminated section at the beginning of the series (default is 0.1).

        seed : bool, optional
            Whether to use a seed for reproducibility (default is True).

        logic_by_series : bool, optional
            Contaminate the series based on the series (sensor) malfunction (default: True).

        explainer : bool, optional
            Only used within the Explainer Module to contaminate one series at a time (default: False).

        verbose : bool, optional
            Whether to display the contamination information (default is True).

        Returns
        -------
        numpy.ndarray
            The contaminated time series data.

        Example
        -------
            >>> ts_m = GenGap.gaussian(ts.data, rate_series=0.2, std_dev=0.4, offset=0.1):

        """
        from scipy.stats import norm

        if logic_by_series:
            input_data = input_data.T # series-based contamination

        ts_contaminated = input_data.copy()
        M, NS = ts_contaminated.shape
        default_init = 0

        if seed:
            seed_value = 42
            np.random.default_rng(seed_value)
            #np.random.seed(seed_value)

        if offset < 1:  # percentage or real value
            offset_nbr = math.ceil(offset * NS)
            if not explainer:
                offset = utils.verification_limitation(offset, low_limit=0)
        else:
            offset_nbr = offset

        values_nbr = int(NS * rate_series)

        if not explainer:  # use random series
            # Validation and limitation of input parameters
            rate_series = utils.verification_limitation(rate_series)
            rate_dataset = utils.verification_limitation(rate_dataset)
            nbr_series_impacted = int(np.ceil(M * rate_dataset))
        else:  # use fix series
            nbr_series_impacted = int(rate_dataset)
            default_init = nbr_series_impacted
            nbr_series_impacted = nbr_series_impacted + 1

        if not explainer and verbose:
            print(f"\n(CONT) missigness pattern: GAUSSIAN"
                  f"\n\tpercentage of contaminated series: {rate_dataset * 100}%"
                  f"\n\trate of missing data per series: {rate_series * 100}%"
                  f"\n\tsecurity offset: [0-{offset_nbr}]"
                  f"\n\tseed value: {seed_value}"
                  f"\n\tmean strategy : {selected_mean}"
                  f"\n\tstandard deviation : {std_dev}\n")

        if offset_nbr + values_nbr > NS:
            raise ValueError(f"\n\tError: The sum of offset ({offset_nbr}) and missing values ({values_nbr}) exceeds the limit of of the series.")


        for series in range(default_init, nbr_series_impacted):
            S = int(series)
            N = len(ts_contaminated[S])  # number of values in the series
            P = GenGap._compute_offset(N=N, offset=offset)  # values to protect in the beginning of the series
            W = int(N * rate_series)  # number of data points to remove
            R = np.arange(P, N)

            # probability density function
            mean = np.mean(ts_contaminated[S])
            mean = max(min(mean, 1), -1)

            if selected_mean == "position":
                center = (P + N) / 2
            else:
                center = P + mean * (N - P)

            scale = std_dev * (N - P)

            probabilities = norm.pdf(R, loc=center, scale=scale)

            # normalizes the probabilities so that their sum equals 1
            probabilities /= probabilities.sum()

            # select the values based on the probability
            missing_indices = np.random.choice(R, size=W, replace=False, p=probabilities)

            # apply missing values
            ts_contaminated[S, missing_indices] = np.nan

        if logic_by_series:
            return ts_contaminated.T
        else:
            return ts_contaminated

    def distribution(input_data, rate_dataset=0.2, rate_series=0.2, probabilities_list=None, offset=0.1, seed=True, logic_by_series=True, explainer=False, verbose=True):
        """
        Missingness follows a probability distribution, each position has a certain chance of being missing.

        Docs: https://imputegap.readthedocs.io/en/latest/missingness_patterns.html


        Parameters
        ----------
        input_data : numpy.ndarray
            The time series dataset to contaminate.

        rate_dataset : float, optional
            Percentage of series to contaminate (default is 0.2).

        rate_series : float, optional
            Percentage of missing values per series (default is 0.2).

        probabilities_list : 2-D array-like, optional
            The probabilities of being contaminated associated with each values of a series.
            Most match the shape of input data without the offset : (e.g. [[0.1, 0, 0.3, 0], [0.2, 0.1, 0.2, 0.9]])

        offset : float, optional
            Size of the uncontaminated section at the beginning of the series (default is 0.1).

        seed : bool, optional
            Whether to use a seed for reproducibility (default is True).

        logic_by_series : bool, optional
            Contaminate the series based on the series (sensor) malfunction (default: True).

        explainer : bool, optional
            Only used within the Explainer Module to contaminate one series at a time (default: False).

        verbose : bool, optional
            Whether to display the contamination information (default is True).

        Returns
        -------
        numpy.ndarray
            The contaminated time series data.

        Example
        -------
            >>> ts_m = GenGap.distribution(ts.data, rate_dataset=0.2, rate_series=0.2, probabilities_list=probabilities_list, offset=0.1)

        """

        if probabilities_list is None:
            print(f"(ERROR) distribution pattern needs a probabilities list as input.\n")
            return input_data

        if logic_by_series:
            input_data = input_data.T # series-based contamination

        ts_contaminated = input_data.copy()
        M, NS = ts_contaminated.shape
        default_init = 0

        if seed:
            seed_value = 42
            np.random.default_rng(seed_value)
            #np.random.seed(seed_value)

        if offset < 1:  # percentage or real value
            offset_nbr = math.ceil(offset * NS)
            if not explainer:
                offset = utils.verification_limitation(offset, low_limit=0)
        else:
            offset_nbr = offset

        values_nbr = int(NS * rate_series)

        if not explainer:  # use random series
            # Validation and limitation of input parameters
            rate_series = utils.verification_limitation(rate_series)
            rate_dataset = utils.verification_limitation(rate_dataset)
            nbr_series_impacted = int(np.ceil(M * rate_dataset))
        else:  # use fix series
            nbr_series_impacted = int(rate_dataset)
            default_init = nbr_series_impacted
            nbr_series_impacted = nbr_series_impacted + 1

        if not explainer and verbose:
            print(f"\n(CONT) missigness pattern: DISTRIBUTION"
                  f"\n\tpercentage of contaminated series: {rate_dataset * 100}%"
                  f"\n\trate of missing data per series: {rate_series * 100}%"
                  f"\n\tsecurity offset: [0-{offset_nbr}]"
                  f"\n\tseed value: {seed_value}"
                  f"\n\tprobabilities list : {np.array(probabilities_list).shape}\n")

        if offset_nbr + values_nbr > NS:
            raise ValueError(f"\n\tError: The sum of offset ({offset_nbr}) and missing values ({values_nbr}) exceeds the limit of of the series.")

        if np.array(probabilities_list).shape != (M, NS - offset_nbr):
            raise ValueError(f"\n\tError: The probability list does not match the matrix in input {np.array(probabilities_list).shape} != ({M},{NS - offset_nbr}).")

        for series in range(default_init, nbr_series_impacted):
            S = int(series)
            N = len(ts_contaminated[S])  # number of values in the series
            P = GenGap._compute_offset(N=N, offset=offset)  # values to protect in the beginning of the series
            W = int(N * rate_series)  # number of data points to remove
            R = np.arange(P, N)
            D = probabilities_list[S]

            missing_indices = np.random.choice(R, size=W, replace=False, p=D)

            # apply missing values
            ts_contaminated[S, missing_indices] = np.nan

        if logic_by_series:
            return ts_contaminated.T
        else:
            return ts_contaminated


    def disjoint(input_data, rate_series=0.1, limit=1, offset=0.1, logic_by_series=True, verbose=True):
        """
        Each missing block begins where the previous one ends, so the missing intervals are consecutive and do not overlap.

        Docs: https://imputegap.readthedocs.io/en/latest/missingness_patterns.html


        Parameters
        ----------
        input_data : numpy.ndarray
            The time series dataset to contaminate.

        rate_series : float, optional
            Percentage of missing values per series (default is 0.1).

        limit : float, optional
            Percentage expressing the limit index of the end of the contamination (default is 1: all length).

        offset : float, optional
            Size of the uncontaminated section at the beginning of the series (default is 0.1).

        logic_by_series : bool, optional
            Contaminate the series based on the series (sensor) malfunction (default: True).

        verbose : bool, optional
            Whether to display the contamination information (default is True).

        Returns
        -------
        numpy.ndarray
            The contaminated time series data.

        Example
        -------
            >>> ts_m = GenGap.disjoint(ts.data, rate_series=0.1, limit=1, offset=0.1)

        """

        if logic_by_series:
            input_data = input_data.T # series-based contamination

        ts_contaminated = input_data.copy()
        M, NS = ts_contaminated.shape

        rate_series = utils.verification_limitation(rate_series)

        if offset < 1:  # percentage or real value
            offset_nbr = math.ceil(offset * NS)
            offset = utils.verification_limitation(offset, low_limit=0)
        else:
            offset_nbr = offset

        values_nbr = int(NS * rate_series)

        if verbose:
            print(f"\n(CONT) missigness pattern: DISJOINT"
                  f"\n\tpercentage of contaminated series: {rate_series * 100}%"
                  f"\n\trate of missing data per series: {rate_series * 100}%"
                  f"\n\tsecurity offset: [0-{offset_nbr}]"
                  f"\n\tlimit: {limit}\n")

        if offset_nbr + values_nbr > NS:
            raise ValueError(f"\n\tError: The sum of offset ({offset_nbr}) and missing values ({values_nbr}) exceeds the limit of of the series.")

        S = 0
        X = 0
        final_limit = int(NS*limit)-1

        while S < M:
            N = len(ts_contaminated[S])  # number of values in the series
            P = GenGap._compute_offset(N=N, offset=offset)  # values to protect in the beginning of the series
            W = int(N * rate_series)  # number of data to remove
            L = X + W  # new limit

            for to_remove in range(X, L):
                index = P + to_remove
                ts_contaminated[S, index] = np.nan

                if index >= final_limit:  # reach the limitation
                    if logic_by_series:
                        return ts_contaminated.T
                    else:
                        return ts_contaminated

            X = L
            S = S + 1

        if logic_by_series:
            return ts_contaminated.T
        else:
            return ts_contaminated

    def overlap(input_data, rate_series=0.2, limit=1, shift=0.05, offset=0.1, logic_by_series=True, verbose=True):
        """
        Each missing block starts at the end of the previous one with a specified shift, so the missing intervals are consecutive and overlap.

        Docs: https://imputegap.readthedocs.io/en/latest/missingness_patterns.html

        Parameters
        ----------
        input_data : numpy.ndarray
            The time series dataset to contaminate.

        rate_series : float, optional
            Percentage of missing values per series (default is 0.2).

        limit : float, optional
            Percentage expressing the limit index of the end of the contamination (default is 1: all length).

        shift : float, optional
            Percentage of shift inside each the last disjoint contamination.

        offset : float, optional
            Size of the uncontaminated section at the beginning of the series (default is 0.1).

        logic_by_series : bool, optional
            Contaminate the series based on the series (sensor) malfunction (default: True).

        verbose : bool, optional
            Whether to display the contamination information (default is True).

        Returns
        -------
        numpy.ndarray
            The contaminated time series data.

        Example
        -------
            >>> ts_m = GenGap.overlap(ts.data, rate_series=0.1, limit=1, shift=0.05, offset=0.1)

        """

        if logic_by_series:
            input_data = input_data.T # series-based contamination

        ts_contaminated = input_data.copy()
        M, NS = ts_contaminated.shape

        rate_series = utils.verification_limitation(rate_series)

        if offset < 1:  # percentage or real value
            offset_nbr = math.ceil(offset * NS)
            offset = utils.verification_limitation(offset, low_limit=0)
        else:
            offset_nbr = offset

        values_nbr = int(NS * rate_series)

        if verbose:
            print(f"\n(CONT) missigness pattern: OVERLAP"
                  f"\n\tpercentage of contaminated series: {rate_series * 100}%"
                  f"\n\trate of missing data per series: {rate_series * 100}%"
                  f"\n\tsecurity offset: [0-{offset_nbr}]"
                  f"\n\tshift: {shift * 100} %"
                  f"\n\tlimit: {limit}\n")

        if offset_nbr + values_nbr > NS:
            raise ValueError(f"\n\tError: The sum of offset ({offset_nbr}) and missing values ({values_nbr}) exceeds the limit of of the series.")

        if int(NS*shift) > int(NS*offset):
            raise ValueError(f"Shift too big for this dataset and offset: shift ({int(NS*shift)}), offset ({int(NS*offset)}).")

        S, X = 0, 0
        final_limit = int(NS * limit) - 1

        while S < M:
            N = len(ts_contaminated[S])  # number of values in the series
            P = GenGap._compute_offset(N=N, offset=offset)  # values to protect in the beginning of the series
            W = int(N * rate_series)  # number of data to remove

            if X != 0:
                X = X - int(N * shift)

            L = X + W  # new limit

            for to_remove in range(X, L):
                index = P + to_remove
                ts_contaminated[S, index] = np.nan

                if index >= final_limit:  # reach the limitation
                    if logic_by_series:
                        return ts_contaminated.T
                    else:
                        return ts_contaminated

            X = L
            S = S + 1

        if logic_by_series:
            return ts_contaminated.T
        else:
            return ts_contaminated

