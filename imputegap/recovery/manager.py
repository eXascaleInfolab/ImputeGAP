import datetime
import math
import os
import platform
import time
import numpy as np
import matplotlib
import importlib.resources

from imputegap.tools import utils

import matplotlib.pyplot as plt


def select_backend():
    system = platform.system()
    #headless = os.getenv("DISPLAY") is None or os.getenv("CI") is not None
    if system == "Darwin":
        for backend in ["MacOSX", "Qt5Agg", "TkAgg"]:
            try:
                matplotlib.use(backend)
                return
            except (ImportError, RuntimeError):
                continue
        try:
            matplotlib.use("TkAgg")  # fallback
        except (ImportError, RuntimeError):
            matplotlib.use("Agg")
    else:
        for backend in ["TkAgg", "QtAgg", "Qt5Agg", "Agg"]:
            try:
                matplotlib.use(backend)
                return
            except (ImportError, RuntimeError):
                continue



class TimeSeries:
    """
    Class for managing and manipulating time series data.

    This class allows importing, normalizing, and visualizing time series datasets. It also provides methods
    to contaminate the datasets with missing values and plot results.

    Methods
    -------
    __init__() :
        Initializes the TimeSeries object.

    import_matrix(data=None) :
        Imports a matrix of time series data.

    load_series(data, nbr_series=None, nbr_val=None, header=False, normalizer="z_score", replace_nan=False, reverse=False, verbose=True):
        Loads time series data from a file or predefined dataset.

    print(limit=10, view_by_series=False) :
        Prints a limited number of time series from the dataset.

    print_results(metrics, algorithm="") :
        Prints the results of the imputation process.

    normalize(normalizer="z_score", data=None, verbose=True):
        Normalizes the time series dataset.

    plot(input_data, incomp_data=None, recov_data=None, max_series=None, max_values=None, size=(16, 8), save_path="", display=True) :
        Plots the time series data, including raw, contaminated, or imputed data.

    """

    def __init__(self, verbose=True):
        """
        Initialize the TimeSeries object.

        The class works with time series datasets, where each series is separated by space, and values
        are separated by newline characters.

        IMPORT FORMAT : (Values,Series) : series are seperated by "SPACE" et values by "\\n"
        """
        self.data = None
        self.name = "default"
        self.plots = None
        self.algorithms = utils.list_of_algorithms()
        self.patterns = utils.list_of_patterns()
        self.datasets = utils.list_of_datasets()
        self.optimizers = utils.list_of_optimizers()
        self.extractors = utils.list_of_extractors()
        self.forecasting_models = utils.list_of_downstreams()
        self.families = utils.list_of_families()
        self.algorithms_with_families = utils.list_of_algorithms_with_families()
        self.reversed = False
        select_backend()

        if verbose:
            print(f"\nImputeGAP Library has been invoked (https://github.com/eXascaleInfolab/ImputeGAP)\n")

    def import_matrix(self, data=None):
        """
        Imports a matrix of time series data.

        The data can be provided as a list or a NumPy array. The format is (Series, Values),
        where series are separated by space, and values are separated by newline characters.

        Parameters
        ----------
        data : list or numpy.ndarray, optional
            The matrix of time series data to import.

        Returns
        -------
        TimeSeries
            The TimeSeries object with the imported data.
        """
        if data is not None:
            if isinstance(data, list):
                self.data = np.array(data)

            elif isinstance(data, np.ndarray):
                self.data = data
            else:
                print("\nThe time series have not been loaded, format unknown\n")
                self.data = None
                raise ValueError("Invalid input for import_matrix")

            return self

    def load_series(self, data, nbr_series=None, nbr_val=None, header=False, normalizer="z_score", replace_nan=False, reverse=False, verbose=True):
        """
        Loads time series data from a file or predefined dataset.

        The data is loaded as a matrix of shape (Values, Series). You can limit the number of series
        or values per series for computational efficiency.

        Parameters
        ----------
        data : str
            The file path or name of a predefined dataset (e.g., 'bafu.txt').

        nbr_series : int, optional
            The maximum number of series to load.

        nbr_val : int, optional
            The maximum number of values per series.

        header : bool, optional
            Whether the dataset has a header. Default is False.

        normalizer : str, optional
            The normalization technique to use. Options are "z_score" or "min_max". Default is "z_score".
            To keep the raw data, set normalizer to None | normalizer=None

        replace_nan : bool, optional
            The Dataset has already NaN values that needs to be replaced by 0 values.

        reverse: bool, optional
            Order of the 1st dimension of the dataset, series or values/timestamps. Default is False
            e.g. True : (50, 1000) / 50 sensors (lines) of 10000 values/timestamps (cols)
            e.g. False : (1000, 50) / 1000 values/timestamps (lines) for 50 sensors (cols)

        verbose : bool, optional
            Display information print (default: True).


        Returns
        -------
        TimeSeries
            The TimeSeries object with the loaded data.

        Example
        -------
            >>> ts.load_series(utils.search_path("eeg-alcohol"), nbr_series=50, nbr_val=100)

        """

        if data is not None:
            if isinstance(data, str):
                saved_data = data

                #  update path form inner library datasets
                if data in utils.list_of_datasets(txt=True):
                    self.name = data[:-4]
                    data = importlib.resources.files('imputegap.datasets').joinpath(data)

                if not os.path.exists(data):
                    here = os.path.dirname(os.path.dirname(__file__))
                    data = os.path.join(here, "datasets/", data)

                self.data = np.genfromtxt(data, delimiter=' ', max_rows=nbr_val, skip_header=int(header))

                if verbose:
                    print("\n(SYS) The dataset is loaded from " + str(data) + "\n")

                if nbr_series is not None:
                    self.data = self.data[:, :nbr_series]
            else:
                print("\nThe dataset has not been loaded, format unknown\n")
                self.data = None
                raise ValueError("Invalid input for load_series")

            if replace_nan:
                print("\nThe NaN values has been set to zero...\n")
                self.data = np.nan_to_num(self.data)  # Replace NaNs with 0

            self.reversed = reverse

            if self.reversed:
                self.data = self.data.T
            else:
                self.data = self.data

            if normalizer is not None:
                self.data = self.normalize(normalizer=normalizer, data=self.data, verbose=verbose)

            return self

    def print(self, nbr_val=10, nbr_series=7, view_by_series=True):
        """
        Prints a limited number of time series from the dataset.

        Parameters
        ----------
        nbr_val : int, optional
        The number of timestamps to print. Default is 15. Use -1 for no restriction.
        nbr_series : int, optional
        The number of series to print. Default is 10. Use -1 for no restriction.
        view_by_series : bool, optional
        Whether to view by series (True) or by values (False).

        Returns
        -------
        None
        """
        to_print = self.data
        nbr_tot_values, nbr_tot_series = to_print.shape
        print_col, print_row = "series", "timestamp"
        print_col_inc, print_row_inc = 0, 1

        print(f"\nshape of {self.name} : {self.data.shape}\n\tnumber of series\t\t= {nbr_tot_series}\n\tnumber of timestamps\t= {nbr_tot_values}\n")

        if nbr_val == -1:
            nbr_val = to_print.shape[1]
        if nbr_series == -1:
            nbr_series = to_print.shape[0]
        to_print = to_print[:nbr_series, :nbr_val]

        if not view_by_series:
            to_print = to_print.T
            print_col, print_row = "timestamp", "series"
            print_col_inc, print_row_inc = 1, 0

        header_format = "{:<15}"  # Fixed size for headers
        value_format = "{:>15.10f}"  # Fixed size for values
        # Print the header
        print(f"{'':<18}", end="")  # Empty space for the row labels
        for i in range(to_print.shape[1]):
            print(header_format.format(f"{print_col}_{i + print_col_inc}"), end="")
        print()

        # Print each limited series with fixed size
        for i, series in enumerate(to_print):
            print(header_format.format(f"{print_row}_{i + print_row_inc}"), end="")
            print("".join([value_format.format(elem) for elem in series]))

        if nbr_series < nbr_tot_series:
            print("...")

    def print_results(self, metrics, algorithm="", text="Results"):
        """
        Prints the results of the imputation process.

        Parameters
        ----------
        metrics : dict
           A dictionary containing the imputation metrics to display.
        algorithm : str, optional
           The name of the algorithm used for imputation.
        algorithm : str, optional
           Output text to help the user.

        Returns
        -------
        None

        Example
        -------
            >>> ts.print_results(imputer.metrics, imputer.algorithm)
        """

        if algorithm != "":
            print(f"\n{text} ({algorithm}) :")
        else:
            print(f"\n{text} :")

        for key, value in metrics.items():
            print(f"{key:<20} = {value}")

    def normalize(self, normalizer="z_score", data=None, verbose=True):
        """
        Normalize the time series dataset.

        Supported normalization techniques are "z_score" and "min_max". The method also logs
        the execution time for the normalization process.

        Parameters
        ----------
        normalizer : str, optional
            The normalization technique to use. Options are "z_score" or "min_max". Default is "z_score".

        data : darray, optional
            Matrix to normalize (outside of the object).

        verbose : bool, optional
            Whether to display the contamination information (default is False).

        Returns
        -------
        numpy.ndarray
            The normalized time series data.

        Example
        -------
            >>> ts.normalize(normalizer="z_score")
        """

        normalizer = normalizer.replace("-", "_").lower()

        if data is not None:
            self.data = data

        if self.reversed:
            self.data = self.data.T

        if normalizer == "min_max":
            start_time = time.time()  # Record start time

            # Compute the min and max for each series (column-wise), ignoring NaN
            ts_min = np.nanmin(self.data, axis=0)
            ts_max = np.nanmax(self.data, axis=0)

            # Compute the range for each series, and handle cases where the range is 0
            range_ts = ts_max - ts_min
            range_ts[range_ts == 0] = 1  # Prevent division by zero for constant series

            # Apply min-max normalization
            self.data = (self.data - ts_min) / range_ts

            end_time = time.time()
        elif normalizer == "z_lib":
            from scipy.stats import zscore

            start_time = time.time()  # Record start time

            self.data = zscore(self.data, axis=0)

            end_time = time.time()

        elif normalizer == "m_lib":
            from sklearn.preprocessing import MinMaxScaler

            start_time = time.time()  # Record start time

            scaler = MinMaxScaler()
            self.data = scaler.fit_transform(self.data)

            end_time = time.time()
        elif normalizer == "z_score":
            start_time = time.time()  # Record start time

            mean = np.nanmean(self.data, axis=0)
            std_dev = np.nanstd(self.data, axis=0)

            # Avoid division by zero: set std_dev to 1 where it is zero
            std_dev[std_dev == 0] = 1

            # Apply z-score normalization
            self.data = (self.data - mean) / std_dev

            end_time = time.time()
        else:
            start_time = time.time()
            if verbose:
                print(f"> (ERROR): normalizer not recognised...")
            end_time = time.time()

        if self.reversed:
            self.data = self.data.T

        if verbose:
            print(f"> logs: normalization ({normalizer}) of the data - runtime: {(end_time - start_time):.4f} seconds")

        if data is not None:
            return self.data

    def plot(self, input_data, incomp_data=None, recov_data=None, nbr_series=None, nbr_val=None, series_range=None, subplot=False, size=(16, 8), algorithm=None, save_path="./imputegap_assets", style="default", cont_rate=None, grid=True, reverse=True, legends=True, display=True, verbose=True):
        """
        Plot the time series data, including raw, contaminated, or imputed data.

        Parameters
        ----------
        input_data : numpy.ndarray
            The original time series data without contamination.

        incomp_data : numpy.ndarray, optional
            The contaminated time series data.

        recov_data : numpy.ndarray, optional
            The imputed time series data.

        nbr_series : int, optional
            The maximum number of series to plot.

        nbr_val : int, optional
            The maximum number of values per series to plot.

        series_range : int, optional
            The index of a specific series to plot. If set, only this series will be plotted.

        subplot : bool, optional
            Print one time series by subplot or all in the same plot.

        size : tuple, optional
            Size of the plot in inches. Default is (16, 8).

        algorithm : str, optional
            Name of the algorithm used for imputation.

        save_path : str, optional
            Path to save the plot locally.

        style : str, optional
            Name of the style used for the plot ("default" / "mono": specific series more visible).

        cont_rate : str, optional
            Percentage of contamination in each series to plot.

        grid : bool, optional
            Whether to plot in a grid or not.

        reverse : bool, optional
            Reverse the plot to see timestamps as x axis and values as y axis.

        legends: bool, optional
            Display or not the legend in the plot (default is True).

        display : bool, optional
            Whether to display the plot. Default is True.

        verbose : bool, optional
            Whether to display the plot information. Default is True.

        Returns
        -------
        str or None
            The file path of the saved plot, if applicable.

        Example
        -------
            >>> ts.plot(input_data=ts.data, nbr_series=9, nbr_val=100, save_path="./imputegap_assets") # plain data
            >>> ts.plot(ts.data, ts_m, nbr_series=9, subplot=True, save_path="./imputegap_assets") # contamination
            >>> ts.plot(input_data=ts.data, incomp_data=ts_m, recov_data=imputer.recov_data, nbr_series=9, subplot=True, save_path="./imputegap_assets") # imputation
        """
        select_backend()

        if reverse:
            input_data = input_data.T
            if incomp_data is not None:
                incomp_data = incomp_data.T
            if recov_data is not None:
                recov_data = recov_data.T

        number_of_series = 0
        if algorithm is None:
            algorithm = "imputegap"
            title_imputation = "Imputed Data"
            title_contamination = "Missing Data"
        else:
            title_imputation = algorithm.lower()
            title_contamination = algorithm.lower()

        if nbr_series is None or nbr_series == -1:
            nbr_series = input_data.shape[0]
        if nbr_val is None or nbr_val == -1:
            nbr_val = input_data.shape[1]

        if subplot:
            series_indices = [i for i in range(incomp_data.shape[0]) if np.isnan(incomp_data[i]).any()]
            count_series = [series_range] if series_range is not None else range(min(len(series_indices), nbr_series))
            n_series_to_plot = len(count_series)
        else:
            series_indices = [series_range] if series_range is not None else range(min(input_data.shape[0], nbr_series))
            n_series_to_plot = len(series_indices)

        if n_series_to_plot == 0:
            n_series_to_plot = min(nbr_series, incomp_data.shape[0])

        if subplot:
            n_cols = min(3, n_series_to_plot)
            n_rows = (n_series_to_plot + n_cols - 1) // n_cols

            x_size, y_size = size
            x_size = x_size * n_cols
            y_size = y_size * n_rows

            scale_factor = 0.85
            x_size_screen = (1920 / 100) * scale_factor
            y_size_screen = (1080 / 100) * scale_factor

            if n_rows < 4:
                x_size = x_size_screen
                y_size = y_size_screen

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(x_size, y_size), squeeze=False)
            fig.canvas.manager.set_window_title(algorithm)
            axes = axes.flatten()
        else:
            plt.figure(figsize=size)

            if grid:
                plt.grid(grid, linestyle='--', color='#d3d3d3', linewidth=0.6)

        if input_data is not None:
            if style == "default":
                colors = utils.load_parameters("default", algorithm="colors", verbose=False)
            else:
                colors = utils.load_parameters("default", algorithm="colors_blacks", verbose=False)

            if nbr_series == 1:
                colors = ["blue"]

            for idx, i in enumerate(series_indices):

                if subplot:
                    color = colors[0]
                else:
                    color = colors[i % len(colors)]

                timestamps = np.arange(min(input_data.shape[1], nbr_val))

                # Select the current axes if using subplots
                if subplot:
                    ax = axes[idx]

                    if grid:
                        ax.grid(grid, linestyle='--', color='#d3d3d3', linewidth=0.6)
                else:
                    ax = plt

                if incomp_data is None and recov_data is None:  # plot only raw matrix
                    ax.plot(timestamps, input_data[i, :nbr_val], linewidth=2.5, color=color, linestyle='-', label=f'Series_' + str(i+1))

                if incomp_data is not None and recov_data is None:  # plot infected matrix
                    if np.isnan(incomp_data[i, :]).any():
                        if style == "default":
                            ax.plot(timestamps, input_data[i, :nbr_val], linewidth=2, color=color, linestyle=':', label=title_contamination)
                        else:
                            ax.plot(timestamps, input_data[i, :nbr_val], linewidth=2, color="red", linestyle='--', label=title_contamination)


                    if np.isnan(incomp_data[i, :]).any() or not subplot:
                        if style == "default":
                            ax.plot(np.arange(min(incomp_data.shape[1], nbr_val)), incomp_data[i, :nbr_val], color=color, linewidth=2.5, linestyle='-', label=f'Series_' + str(i+1))
                        else:
                            ax.plot(np.arange(min(incomp_data.shape[1], nbr_val)), incomp_data[i, :nbr_val], color=color, linewidth=7, linestyle='-', label=f'Series_' + str(i+1))

                if recov_data is not None:  # plot imputed matrix
                    if np.isnan(incomp_data[i, :]).any():
                        ax.plot(np.arange(min(recov_data.shape[1], nbr_val)), recov_data[i, :nbr_val], linewidth=1.5, linestyle='-', color="r", label=title_imputation)

                        ax.plot(timestamps, input_data[i, :nbr_val], linewidth=1.5, linestyle=':', color=color, label=f'Missing Data')

                    if np.isnan(incomp_data[i, :]).any() or not subplot:
                        ax.plot(np.arange(min(incomp_data.shape[1], nbr_val)), incomp_data[i, :nbr_val], color=color, linewidth=2.5, linestyle='-', label=f'Series_' + str(i+1))

                # Label and legend for subplot
                if subplot:
                    handles, labels = ax.get_legend_handles_labels()

                    ax.set_title('Series ' + str(i+1), fontsize=9)
                    #ax.plot([], [], ' ', label='Series ' + str(i + 1))  # invisible line with label
                    ax.set_xlabel('Timestamps', fontsize=7)
                    ax.set_ylabel('Values', fontsize=7)
                    if legends:
                        ax.legend(handles[::-1], labels[::-1], loc='upper left', fontsize=6, frameon=True, fancybox=True, framealpha=0.8, ncol=len(ax.get_legend_handles_labels()[0]))
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    fig.subplots_adjust(top=0.96, hspace=0.4)
                else:
                    plt.tight_layout(rect=[0.01, 0.03, 0.88, 0.95])

                number_of_series += 1
                if number_of_series == nbr_series:
                    break

        if subplot:
            for idx in range(len(series_indices), len(axes)):
                axes[idx].axis('off')

        if not subplot:
            ax = plt.gca()
            ax.tick_params(axis='both', labelsize=14)  # increase tick label size

            plt.xlabel('Timestamps', fontsize=15)
            plt.ylabel('Values', fontsize=15)
            if legends:
                plt.legend(
                    loc='upper left',
                    fontsize=9,
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                    borderpad=1.5,
                    bbox_to_anchor=(1.02, 1),  # Adjusted to keep the legend inside the window
                )

        file_path = None

        if save_path:
            os.makedirs(save_path, exist_ok=True)

            now = datetime.datetime.now()
            current_time = now.strftime("%y_%m_%d_%H_%M_%S")

            if not legends:
                current_time = "imputegap"

            if cont_rate is None:
                file_path = os.path.join(save_path + "/" + current_time + "_" + algorithm + "_plot.jpg")
            else:
                file_path = os.path.join(save_path + "/" + cont_rate + "_" + algorithm + "_plot.jpg")

            plt.savefig(file_path, bbox_inches='tight')

            if verbose:
                print("\nplots saved in:", file_path)

        if display:
            plt.show()

        self.plots = plt

        return file_path


    def shift(self, id_series, shift_value=0.01):
       """
       Shift the values of a series.

       Parameters
       ----------
       id_series : int
           The index of the series to shift

       shift_value : float
           Values of shift (vertically) (default: 0.01).
       """
       if self.data.shape[0] > id_series > 0:
           self.data[:, id_series] += shift_value
           print(f"(SYS) Time series {id_series} data as been shift by: {shift_value}")
       else:
           print(f"(ERR) The series {id_series} has no data.")


    def range(self, starting_series, ending_series):
        """
       Select a subset of series from the dataset within a given range.

       Parameters
       ----------
       starting_series : int
           The index of the first series to keep (inclusive).

       ending_series : int
           The index of the last series to keep (inclusive).
       """
        if self.data.shape[0] > starting_series > 0:
            if self.data.shape[0] > ending_series > 0:
                self.data = self.data[starting_series:ending_series + 1]
                print(f"(SYS) Time series data as been rescaled: {self.data.shape}")
            else:
                print(f"(ERR) The series {starting_series} has no data.")
        else:
            print(f"(ERR) The series {ending_series} has no data.")
