# Standard library
import datetime
import os
import math
import time
import re

# Third-party libraries
import pandas as pd
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import itertools
from pathlib import Path

# ImputeGAP imports
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries

# Global Variable
MAX_PX = 60000  # Matplotlib hard limit (per dimension)

class Benchmark:
    """
    A class to evaluate the performance of imputation algorithms through benchmarking across datasets and patterns.

    Methods
    -------
    average_runs_by_names(self, data):
        Average the results of all runs depending on the dataset.
    avg_results_task():
        Calculate average metrics (e.g., RMSE) across multiple datasets and algorithm runs.
    generate_heatmap():
        Generate and save a heatmap visualization of RMSE scores for datasets and algorithms.
    generate_reports_txt():
        Create detailed text-based reports summarizing metrics and timing results for all evaluations.
    generate_reports_excel():
        Create detailed excel-based reports summarizing metrics and timing results for all evaluations.
    generate_plots():
        Visualize metrics (e.g., RMSE, MAE) and timing (e.g., imputation, optimization) across patterns and datasets.
    eval():
        Perform a complete benchmarking pipeline, including contamination, imputation, evaluation, and reporting.

    Example
    -------
    output : {'eegalcohol': {'mcar': {'MeanImpute': {'default_params': {'0.05': {'scores': {'RMSE': 1.107394798606378, 'MAE': 0.9036474830477748, 'CORRELATION': nan, 'RUNTIME': 10.07390022277832, 'RUNTIME_LOG': 1.00319764506136}}, '0.1': {'scores': {'RMSE': 0.8569349076796438, 'MAE': 0.6416542359734557, 'CORRELATION': nan, 'RUNTIME': 1.0, 'RUNTIME_LOG': 0.0}}, '0.2': {'scores': {'RMSE': 0.9924113085421721, 'MAE': 0.7939689811173046, 'CORRELATION': nan, 'RUNTIME': 1.0, 'RUNTIME_LOG': 0.0}}, '0.4': {'scores': {'RMSE': 1.0058063455061463, 'MAE': 0.8076546785476064, 'CORRELATION': nan, 'RUNTIME': 1.0, 'RUNTIME_LOG': 0.0}}, '0.6': {'scores': {'RMSE': 0.9891809506243663, 'MAE': 0.7914550709031675, 'CORRELATION': nan, 'RUNTIME': 1.0, 'RUNTIME_LOG': 0.0}}, '0.8': {'scores': {'RMSE': 0.9927953862507292, 'MAE': 0.7925635744718286, 'CORRELATION': nan, 'RUNTIME': 1.0, 'RUNTIME_LOG': 0.0}}}}, 'SoftImpute': {'default_params': {'0.05': {'scores': {'RMSE': 0.4359915238078244, 'MAE': 0.3725965559420608, 'CORRELATION': 0.9530448037164908, 'RUNTIME': 199.30577278137207, 'RUNTIME_LOG': 2.2995198779819055}}, '0.1': {'scores': {'RMSE': 0.3665001858394363, 'MAE': 0.2989983612840734, 'CORRELATION': 0.9049909722894052, 'RUNTIME': 117.54822731018066, 'RUNTIME_LOG': 2.0702160841184516}}, '0.2': {'scores': {'RMSE': 0.39833006221984, 'MAE': 0.30824644022807457, 'CORRELATION': 0.9161465703422209, 'RUNTIME': 317.5652027130127, 'RUNTIME_LOG': 2.5018329084349737}}, '0.4': {'scores': {'RMSE': 0.435591016228979, 'MAE': 0.3335144215651955, 'CORRELATION': 0.9021032587324183, 'RUNTIME': 302.2916316986084, 'RUNTIME_LOG': 2.4804261248244566}}, '0.6': {'scores': {'RMSE': 0.4500113661547204, 'MAE': 0.338085865703361, 'CORRELATION': 0.8893263437029546, 'RUNTIME': 314.93282318115234, 'RUNTIME_LOG': 2.498217926383076}}, '0.8': {'scores': {'RMSE': 0.46554422402146944, 'MAE': 0.3508926604243284, 'CORRELATION': 0.8791443563129441, 'RUNTIME': 311.9697570800781, 'RUNTIME_LOG': 2.4941124947560986}}}}}}}
    """

    def __init__(self):
        """
        Initialize the Benchmark object.
        """
        self.list_results = None
        self.aggregate_results = None
        self.heatmap = None
        self.subplots = None

    def _benchmark_error(self, code, e, runs_plots_scores, ts_m, classifiers, dataset, algorithm, pattern, x, save_dir,
                         verbose):

        print(f"\t\tbypass benchmark: {dataset}, {algorithm}, {pattern}, {x}")

        if e is None:
            e = "by pass errors..."
        dataset_s = dataset
        model = classifiers[0]
        if "-" in dataset:
            dataset_s = dataset.replace("-", "")

        if code == "error":
            print(f"Error during benchmark for {algorithm}, with {dataset_s}, and {x}%: {e}")
            if ts_m is None:
                ts_m = np.zeros(1)
        imputer_tr = utils.config_impute_algorithm(incomp_data=ts_m, algorithm=algorithm, verbose=verbose)

        imputer_tr.metrics = {
            "RMSE": np.nan,
            "MAE": np.nan,
            "MI": np.nan,
            "CORRELATION": np.nan,
            "RUNTIME": np.nan,
            "RUNTIME_LOG": np.nan,
            "DOWNSTREAM_ACC": np.nan,
        }

        runs_plots_scores.setdefault(str(dataset_s), {}).setdefault(str(pattern), {}).setdefault(str(algorithm),
                                                                                                 {}).setdefault(
            str(model), {})[str(x)] = {"scores": imputer_tr.metrics}

        if code == "error":
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "error.log")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                if isinstance(ts_m, TimeSeries):
                    x = None
                else:
                    x = ts_m.shape

                with open(save_path, "a") as file:
                    file.write(
                        f"{timestamp} | Error during benchmark for {algorithm}, with {dataset_s} - for a shape of ({x}) - ({pattern}/{model}), and {x}%: {e}\n\n")
                return imputer_tr, runs_plots_scores

            except Exception as e:
                print(f"Error during the log generation of errors files...: {e}")
                return imputer_tr, runs_plots_scores

        return imputer_tr, runs_plots_scores

    #def _benchmark_exception(self, data, algorithm, pattern, x, N, F):
        #return False

        #if algorithm.upper() == 'DEEPMVI' or algorithm.upper() == 'DEEP_MVI':
        #    if pattern.lower() == "mcar" or pattern.lower() == "missing_completely_at_random":
        #        if x > 0.6:
        #            print(
        #                 f"\n(BENCH) The imputation algorithm {algorithm} is not compatible with this configuration {pattern} with missingness rate more than 0.6.")
        #            return True
        #    if pattern.lower() == "mp" or pattern.lower() == "aligned":
        #        if x < 0.15:
        #           print(
        #                f"\n(BENCH) The imputation algorithm {algorithm} is not compatible with this configuration {pattern} with missingness rate less then 0.15.")
        #            return True
        #    if data == "meteo":
        #        return True
        #
        #if data == "meteo":
        #    if x >= 0.8:
        #        print(
        #            f"\n(BENCH) The imputation algorithm {algorithm} is not compatible with this configuration {data}. Not enough series to train the model.")
        #       return True
        #
        #
        #return False

    def _config_optimization(self, opti_mean, ts_test, pattern, algorithm, block_size_mcar):
        """
        Configure and execute optimization for selected imputation algorithm and pattern.

        Parameters
        ----------
        opti_mean : float
            Mean parameter for contamination.
        ts_test : TimeSeries
            TimeSeries object containing dataset.
        pattern : str
            Type of contamination pattern (e.g., "mcar", "mp", "blackout", "disjoint", "overlap", "gaussian").
        algorithm : str
            Imputation algorithm to use.
        block_size_mcar : int
            Size of blocks removed in MCAR

        Returns
        -------
        BaseImputer
            Configured imputer instance with optimal parameters.
        """

        incomp_data = utils.config_contamination(ts=ts_test, pattern=pattern, dataset_rate=opti_mean,
                                                 series_rate=opti_mean, block_size=block_size_mcar)
        imputer = utils.config_impute_algorithm(incomp_data=incomp_data, algorithm=algorithm)

        return imputer

    def average_runs_by_names(self, data):
        """
        Average the results of all runs depending on the dataset

        Parameters
        ----------
        data : list
            list of dictionary containing the results of the benchmark runs.

        Returns
        -------
        list
            list of dictionary containing the results of the benchmark runs averaged by datasets.
        """
        results_avg, all_names = [], []

        # Extract dataset names
        for dictionary in data:
            all_keys = list(dictionary.keys())
            dataset_name = all_keys[0]
            all_names.append(dataset_name)

        # Get unique dataset names
        unique_names = sorted(set(all_names))

        # Initialize and populate the split matrix
        split = [[0 for _ in range(all_names.count(name))] for name in unique_names]
        for i, name in enumerate(unique_names):
            x = 0
            for y, match in enumerate(all_names):
                if name == match:
                    split[i][x] = data[y]
                    x += 1

        # Iterate over the split matrix to calculate averages
        for datasets in split:
            tmp = [dataset for dataset in datasets if dataset != 0]
            merged_dict = {}
            count = len(tmp)

            # Process and calculate averages
            for dataset in tmp:
                for outer_key, outer_value in dataset.items():
                    for middle_key, middle_value in outer_value.items():
                        for mean_key, mean_value in middle_value.items():
                            for method_key, method_value in mean_value.items():
                                for level_key, level_value in method_value.items():
                                    # Initialize scores and times if not already initialized
                                    merger = merged_dict.setdefault(outer_key, {}
                                                                    ).setdefault(middle_key, {}).setdefault(mean_key, {}
                                                                                                            ).setdefault(
                                        method_key, {}).setdefault(level_key, {"scores": {}})

                                    # Add scores and times
                                    for score_key, v in level_value["scores"].items():
                                        if v is None:
                                            v = 0
                                        merger["scores"][score_key] = (merger["scores"].get(score_key, 0) + v / count)

            results_avg.append(merged_dict)

        return results_avg

    def avg_results_task(self, *datasets, metric="RMSE", upstream=True):
        """
        Calculate the average of all metrics and times across multiple datasets.

        Parameters
        ----------
        datasets : dict
            Multiple dataset dictionaries to be averaged.
        metric : str
            Metric to group.

        Returns
        -------
        List
            Matrix with averaged scores and times for all levels, list of algorithms, list of datasets
        """

        # Step 1: Compute average RMSE across runs for each dataset and algorithm
        aggregated_data = {}

        for runs in datasets:
            for dataset, dataset_items in runs.items():
                if dataset not in aggregated_data:
                    aggregated_data[dataset] = {}

                for pattern, pattern_items in dataset_items.items():
                    for algo, algo_data in pattern_items.items():
                        for m, m_items in algo_data.items():

                            group_key = algo if upstream else m
                            aggregated_data[dataset].setdefault(group_key, [])

                            # if algo not in aggregated_data[dataset]:
                            #    aggregated_data[dataset][algo] = []
                            for missing_values, missing_values_item in m_items.items():
                                for param, param_data in missing_values_item.items():
                                    # rmse = param_data[metric]
                                    val = param_data.get(metric, np.nan)
                                    aggregated_data[dataset][group_key].append(val)

        # Step 2: Compute averages using NumPy
        average_rmse_matrix = {}
        for dataset, groups in aggregated_data.items():
            average_rmse_matrix[dataset] = {}
            for group_key, rmse_values in groups.items():
                rmse_array = np.array(rmse_values)
                avg_rmse = np.mean(rmse_array)
                average_rmse_matrix[dataset][group_key] = avg_rmse

        # Step 3: Create a matrix representation of datasets and algorithms
        datasets_list = list(average_rmse_matrix.keys())
        group_keys = {k for groups in average_rmse_matrix.values() for k in groups.keys()}
        group_keys_list = sorted(group_keys)

        # Prepare a NumPy matrix
        comprehensive_matrix = np.zeros((len(datasets_list), len(group_keys_list)))

        for i, dataset in enumerate(datasets_list):
            for j, key in enumerate(group_keys_list):
                comprehensive_matrix[i, j] = average_rmse_matrix[dataset].get(key, np.nan)

        return comprehensive_matrix, group_keys_list, datasets_list

    def generate_heatmap(self, scores_list, algos, sets, metric="RMSE", save_dir="./reports", display=True):
        """
        Generate and save RMSE matrix in HD quality.

        Parameters
        ----------
        scores_list : np.ndarray
            2D numpy array containing RMSE values.
        algos : list of str
            List of algorithm names (columns of the heatmap).
        sets : list of str
            List of dataset names (rows of the heatmap).
        metric : str, optional
            metric to extract
        save_dir : str, optional
            Directory to save the generated plot (default is "./reports").
        display : bool, optional
            Display or not the plot

        Returns
        -------
        Bool
            True if the matrix has been generated
        """
        save_dir = save_dir + "/_heatmaps/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        nbr_algorithms = len(algos)
        nbr_datasets = len(sets)

        cell_size = 4.0
        x_size = cell_size * nbr_algorithms
        y_size = cell_size * nbr_datasets

        fig, ax = plt.subplots(figsize=(x_size, y_size))
        fig.canvas.manager.set_window_title("benchmark heatmap, " + metric)

        import matplotlib.colors as mcolors
        cmap = mcolors.LinearSegmentedColormap.from_list(f"trunc({plt.cm.Greys.name},{0.3:.2f},{0.9:.2f})",
                                                         plt.cm.Greys(np.linspace(0.3, 0.9, 256)))

        norm_ranges = {"RMSE": (0, 2), "CORRELATION": (-2, 2), "MAE": (0, 1.5), "MI": (-1, 1.5), "runtime": (0, 5000),
                       "runtime_log": (-2, 10), }

        key = metric if metric in norm_ranges else metric.lower()
        vmin, vmax = norm_ranges.get(key, (0, 2000))
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        # Create the heatmap
        heatmap = ax.imshow(scores_list, cmap=cmap, norm=norm, aspect='auto')

        # Add color bar for reference
        cbar = plt.colorbar(heatmap, ax=ax, orientation='vertical')
        cbar.set_label(metric, rotation=270, labelpad=15)

        # Set the tick labels
        ax.set_xticks(np.arange(nbr_algorithms))
        ax.set_xticklabels(algos)
        ax.set_yticks(np.arange(nbr_datasets))
        ax.set_yticklabels(sets)

        # Add titles and labels
        ax.set_title('ImputeGAP Algorithms Comparison')
        ax.set_xlabel('Algorithms')
        ax.set_ylabel('Datasets')

        # Show values on the heatmap
        for i in range(len(sets)):
            for j in range(len(algos)):
                ax.text(j, i, f"{scores_list[i, j]:.2f}",
                        ha='center', va='center',
                        color="black" if scores_list[i, j] < 1 else "white")  # for visibility

        filename = "benchmarking_" + metric.lower() + ".jpg"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')  # Save in HD with tight layout

        # Show the plot
        if display:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

        return True

    def generate_reports_summary(self, run_of_values, save_dir="./reports", dataset="", metrics=None, run=-1, rt=0, title="", upstream=True, verbose=True, ):
        """
        Generate and save a summary report of imputation and downstream evaluation results.

        Parameters
        ----------
        run_of_values : list of dict
            List of dictionaries containing evaluation scores organized by dataset, missingness pattern, algorithm, optimizer, and missing rate.
        save_dir : str, optional
            Directory to save the generated report (default is "./reports").
        dataset : str, optional
            Dataset name (default is "").
        metrics : list of str, optional
            List of evaluation metrics to include in the report (If None, default is ["RMSE"]).
        run : int, optional
            Run number to include in the report (default is -1).
        rt : float, optional
            Total execution runtime in milliseconds (default is 0).
        title : str, optional
            Title used to generate the report filename (default is "").
        upstream : bool, optional
            If True, generate tables with imputation algorithms as columns.
            If False, generate tables with downstream models as columns,
            grouped by imputation algorithm (default is True).
        verbose : bool, optional
            Display selected evaluation results in the console (default is True).

        Returns
        -------
        None
            The report is saved as a log file in the specified directory.
        """

        from collections import defaultdict

        if metrics is None:
            metrics = ["RMSE"]

        os.makedirs(save_dir, exist_ok=True)
        metric_unit = "ms"

        # what to print to console
        if "RMSE" not in metrics:
            to_call = [metrics[0], "RUNTIME"]
            if "f1_imputer" in metrics:
                to_call = ["f1_imputer", "RUNTIME"]
            elif "smape_groundtruth" in metrics:
                to_call = ["smape_imputer", "RUNTIME"]
        else:
            if "f1_imputer" in metrics:
                to_call = [
                    "f1_imputer", "f1_groundtruth", "f1_meanimpute",
                    "accuracy_groundtruth", "accuracy_imputer", "accuracy_meanimpute",
                    "recall_groundtruth", "recall_imputer", "recall_meanimpute",
                    "RMSE", "RUNTIME",
                ]
            elif "smape_groundtruth" in metrics:
                to_call = [
                    "smape_imputer", "smape_groundtruth", "smape_baseline",
                    "mse_groundtruth", "mse_imputer", "mse_baseline",
                    "mae_groundtruth", "mae_imputer", "mae_baseline",
                    "RMSE", "RUNTIME",
                ]
            else:
                to_call = ["RMSE", "RUNTIME"]

        # normalize metrics list
        if metrics is None:
            new_metrics = utils.list_of_metrics()
        else:
            new_metrics = np.copy(metrics)
            if "RUNTIME" not in new_metrics:
                new_metrics = np.append(new_metrics, "RUNTIME")
            if "RUNTIME_LOG" not in new_metrics:
                new_metrics = np.append(new_metrics, "RUNTIME_LOG")

        # ---------------------------------------------------------
        # Scan structure: patterns, algorithms, downstream models (optimizers)
        # ---------------------------------------------------------
        opt_any = None
        all_patterns = set()
        patterns_to_algos = defaultdict(set)
        patterns_to_opts = defaultdict(set)

        for scores in run_of_values:
            for ds, patterns_items in scores.items():
                for pattern, algorithm_items in patterns_items.items():
                    all_patterns.add(pattern)
                    for algorithm, optimizer_items in algorithm_items.items():
                        patterns_to_algos[pattern].add(algorithm)
                        for optimizer in optimizer_items.keys():
                            patterns_to_opts[pattern].add(optimizer)
                            if opt_any is None:
                                opt_any = optimizer

        # ---------------------------------------------------------
        # Write report
        # ---------------------------------------------------------
        title_report = "report_" + title + ".log"
        save_path = os.path.join(save_dir, title_report)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(save_path, "w", encoding="utf-8") as file:
            file.write(f"Generated on: {current_time}\n")
            file.write(f"Total runtime: {rt} (ms)\n")
            if run >= 0:
                file.write(f"Run number: {run}\n")
            file.write("=" * 120 + "\n\n")

            # One big section per pattern, then per metric
            for pattern in sorted(all_patterns):
                algos = sorted(patterns_to_algos[pattern])
                opts = sorted(patterns_to_opts[pattern])  # downstream models/optimizers (e.g., arsenal, cboss)

                for metric in new_metrics:
                    # header
                    if upstream:
                        if metric == "RUNTIME":
                            head = "{" + f"{pattern}, {metric}[{metric_unit}]" + "}"
                        else:
                            head = "{" + f"{pattern}, {metric} - {opt_any}" + "}"
                        file.write(head + "\n")

                    # ---------------------------------------------------------
                    # CASE A: upstream=True  -> columns are algorithms (as before)
                    # ---------------------------------------------------------
                    if upstream:
                        row_map = defaultdict(dict)  # (dataset, rate) -> {algo: score_str}

                        for scores in run_of_values:
                            for ds, patterns_items in scores.items():
                                if pattern not in patterns_items:
                                    continue
                                for algorithm, optimizer_items in patterns_items[pattern].items():
                                    for optimizer, x_data_items in optimizer_items.items():
                                        for rate, payload in x_data_items.items():
                                            val = payload.get("scores", {}).get(metric, None)
                                            if val is not None:
                                                row_map[(ds, rate)][algorithm] = f"{val:.10f}"

                        if not row_map:
                            file.write("[no results]\n\n")
                            continue

                        headers = ["Dataset", "Rate"] + list(algos)
                        ds_width = max(12, max((len(ds) for ds, _ in row_map.keys()), default=0) + 2)
                        rate_width = max(6, max((len(str(r)) for _, r in row_map.keys()), default=0) + 2)
                        algo_width = 18
                        col_widths = [ds_width, rate_width] + [algo_width] * len(algos)

                        def fmt_row(vals, col_widths=col_widths):
                            return "".join(f" {str(v):^{w}} " for v, w in zip(vals, col_widths))

                        header_row = fmt_row(headers)
                        sep_row = "-" * len(header_row)

                        file.write(sep_row + "\n")
                        file.write(header_row + "\n")
                        file.write(sep_row + "\n")

                        if verbose and metric in to_call:
                            print("\n" + head)
                            print(sep_row)
                            print(header_row)
                            print(sep_row)

                        def row_key(k):
                            ds, rate = k
                            try:
                                rf = float(rate)
                                return (0, rf, ds)
                            except Exception:
                                return (1, str(rate), ds)

                        for key in sorted(row_map.keys(), key=row_key):
                            ds, rate = key
                            row_vals = [ds, rate] + [row_map[key].get(a, "") for a in algos]
                            line = fmt_row(row_vals)
                            file.write(line + "\n")
                            if verbose and metric in to_call:
                                print(line)

                        file.write(sep_row + "\n\n")
                        if verbose and metric in to_call:
                            print(sep_row + "\n")

                    # ---------------------------------------------------------
                    # CASE B: upstream=False -> columns are downstream models/optimizers
                    # and we print details per algorithm (one table per algorithm).
                    # ---------------------------------------------------------
                    else:
                        # Build one table per algorithm, with columns = opts
                        any_written = False

                        for algorithm in algos:
                            row_map = defaultdict(dict)  # (dataset, rate) -> {opt: score_str}

                            for scores in run_of_values:
                                for ds, patterns_items in scores.items():
                                    if pattern not in patterns_items:
                                        continue
                                    if algorithm not in patterns_items[pattern]:
                                        continue

                                    optimizer_items = patterns_items[pattern][algorithm]
                                    for optimizer, x_data_items in optimizer_items.items():
                                        for rate, payload in x_data_items.items():
                                            val = payload.get("scores", {}).get(metric, None)
                                            if val is not None:
                                                row_map[(ds, rate)][optimizer] = f"{val:.10f}"

                            if not row_map:
                                continue

                            if metric == "RUNTIME":
                                head = "{" + f"{pattern}, {metric}[{metric_unit}], {algorithm}" + "} "
                            else:
                                head = "{" + f"{pattern}, {metric}, {algorithm}" + "} "
                            file.write(head + "\n")

                            any_written = True

                            headers = ["Dataset", "Rate"] + list(opts)
                            ds_width = max(12, max((len(ds) for ds, _ in row_map.keys()), default=0) + 2)
                            rate_width = max(6, max((len(str(r)) for _, r in row_map.keys()), default=0) + 2)
                            opt_width = 18
                            col_widths = [ds_width, rate_width] + [opt_width] * len(opts)

                            def fmt_row(vals, col_widths=col_widths):
                                return "".join(f" {str(v):^{w}} " for v, w in zip(vals, col_widths))

                            header_row = fmt_row(headers)
                            sep_row = "-" * len(header_row)

                            file.write(sep_row + "\n")
                            file.write(header_row + "\n")
                            file.write(sep_row + "\n")

                            if verbose and metric in to_call:
                                print("\n" + head)
                                print(sep_row)
                                print(header_row)
                                print(sep_row)

                            def row_key(k):
                                ds, rate = k
                                try:
                                    rf = float(rate)
                                    return (0, rf, ds)
                                except Exception:
                                    return (1, str(rate), ds)

                            for key in sorted(row_map.keys(), key=row_key):
                                ds, rate = key
                                row_vals = [ds, rate] + [row_map[key].get(o, "") for o in opts]
                                line = fmt_row(row_vals)
                                file.write(line + "\n")
                                if verbose and metric in to_call:
                                    print(line)

                            file.write(sep_row + "\n\n")
                            if verbose and metric in to_call:
                                print(sep_row + "\n")

                        if not any_written:
                            file.write("[no results]\n\n")

                    file.write("\n")  # spacing between metrics

            # dump raw dict(s)
            file.write("Dictionary of Results:\n")
            file.write(str(run_of_values) + "\n")

    def save_into_excel(self, run_of_values, xlsx_path, engine="openpyxl", upstream=True, verbose=True, sep="|"):

        dicts = [run_of_values] if isinstance(run_of_values, dict) else list(run_of_values)

        if verbose:
            print(f"saving pivot results: xlsx_path={xlsx_path}, upstream={upstream}")

        records = []
        metrics_set = set()

        for run_idx, d in enumerate(dicts):
            for dataset, patterns in d.items():
                for pattern, algos in patterns.items():
                    for algorithm, models in algos.items():
                        for model, rates in models.items():
                            for rate, payload in rates.items():
                                scores = payload.get("scores", {})
                                for metric, val in scores.items():
                                    metrics_set.add(str(metric))
                                    records.append({
                                        "dataset": str(dataset),
                                        "pattern": str(pattern),
                                        "rate": str(rate),
                                        "algorithm": str(algorithm),
                                        "model": str(model),
                                        "metric": str(metric),
                                        "value": val,
                                    })

        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(xlsx_path) or ".", exist_ok=True)

        # If nothing, still create a visible sheet
        if df.empty:
            with pd.ExcelWriter(xlsx_path, engine=engine) as writer:
                pd.DataFrame({"info": ["No results found"]}).to_excel(
                    writer, sheet_name="INFO", index=False
                )
            return xlsx_path

        if upstream:
            metrics = sorted(metrics_set)
        else:
            metrics = []
            seen = set()
            for r in records:
                m = r["metric"]
                if m not in seen:
                    seen.add(m)
                    metrics.append(m)

        # Metric ordering and filtering
        if upstream:
            if "RMSE" in metrics:
                metrics = ["RMSE"] + [
                    m for m in metrics
                    if m not in ["RMSE", "DOWNSTREAM_SMAPE"]
                ]
            elif "SMAPE" in metrics:
                metrics = ["SMAPE"] + [m for m in metrics if m != "SMAPE"]
            elif "F1" in metrics:
                metrics = ["F1"] + [m for m in metrics if m != "F1"]

        else:
            # Downstream forecasting -> smape_imputer first
            # Downstream classification -> f1_imputer first
            if "smape_imputer" in metrics:
                metrics = ["smape_imputer"] + [
                    m for m in metrics if m != "smape_imputer"
                ]
            elif "f1_imputer" in metrics:
                metrics = ["f1_imputer"] + [
                    m for m in metrics if m != "f1_imputer"
                ]

        df["value"] = df["value"].astype(str)
        wrote_any_sheet = False

        with pd.ExcelWriter(xlsx_path, engine=engine) as writer:
            for metric in metrics:
                dfx = df[df["metric"] == metric].copy()
                if dfx.empty:
                    continue

                if upstream:
                    pivot = dfx.pivot_table(
                        index=["dataset", "pattern", "rate", "model"],
                        columns=["algorithm"],
                        values="value",
                        aggfunc=lambda s: sep.join(s.astype(str).tolist()),
                        dropna=False,
                    )

                    pivot.columns = [
                        str(goal) for goal in pivot.columns.to_list()
                    ]

                    out = pivot.reset_index()
                    sheet = str(metric)[:31]
                    out.to_excel(writer, sheet_name=sheet, index=False)
                    wrote_any_sheet = True

                else:
                    # Existing downstream view:
                    # rows = dataset / pattern / rate / algorithm
                    # columns = models
                    pivot = dfx.pivot_table(
                        index=["dataset", "pattern", "rate", "algorithm"],
                        columns=["model"],
                        values="value",
                        aggfunc=lambda s: sep.join(s.astype(str).tolist()),
                        dropna=False,
                    )

                    pivot.columns = [
                        str(goal) for goal in pivot.columns.to_list()
                    ]

                    out = pivot.reset_index()
                    sheet = str(metric)[:31]
                    out.to_excel(writer, sheet_name=sheet, index=False)
                    wrote_any_sheet = True

                    # new: extra downstream view
                    # new: rows = dataset / pattern / rate / model
                    # new: columns = algorithms
                    pivot_algorithms = dfx.pivot_table(
                        index=["dataset", "pattern", "rate", "model"],
                        columns=["algorithm"],
                        values="value",
                        aggfunc=lambda s: sep.join(s.astype(str).tolist()),
                        dropna=False,
                    )

                    # new: flatten columns after pivot
                    pivot_algorithms.columns = [
                        str(goal) for goal in pivot_algorithms.columns.to_list()
                    ]

                    # new: create output dataframe for the extra sheet
                    out_algorithms = pivot_algorithms.reset_index()

                    # new: create a second sheet name for the algorithm-on-top view
                    sheet_algorithms = (str(metric)[:24] + "_algos")[:31]

                    # new: write the extra sheet
                    out_algorithms.to_excel(
                        writer,
                        sheet_name=sheet_algorithms,
                        index=False
                    )
                    wrote_any_sheet = True

            # Safety: ensure at least one visible sheet
            if not wrote_any_sheet:
                pd.DataFrame({
                    "info": ["No metric sheets were written (all empty)."]
                }).to_excel(
                    writer,
                    sheet_name="INFO",
                    index=False
                )

        if verbose:
            if upstream:
                print(
                    f"Saved: {xlsx_path} "
                    f"(sheets={len(metrics) if wrote_any_sheet else 1})"
                )
            else:
                # new: downstream now writes two sheets per metric
                print(
                    f"Saved: {xlsx_path} "
                    f"(sheets={2 * len(metrics) if wrote_any_sheet else 1})"
                )

        return xlsx_path

    def generate_reports_txt(self, runs_plots_scores, save_dir="./reports", given_dataset="", metrics=None, run=-1, rt=0, upstream=True, verbose=True):
        """
        Generate and save a text report of metrics and timing for each dataset, algorithm, and pattern.

        Parameters
        ----------
        runs_plots_scores : dict
            Dictionary containing scores and timing information for each dataset, pattern, and algorithm.
        save_dir : str, optional
            Directory to save the reports file (default is "./reports").
        given_dataset : str, optional
            Name of the data for the report name.
        metrics : str, optional
            List of metrics asked for in the report. If metrics is None, then default is ["RMSE"])
        run : int, optional
            Number of the run.
        rt : float, optional
            Total time of the run.
        verbose : bool, optional
            Whether to display the contamination information (default is True).

        Returns
        -------
        None

        Notes
        -----
        The report is saved in a "report.txt" file in `save_dir`, organized in sections with headers and results.
        """
        os.makedirs(save_dir, exist_ok=True)
        metric_unit = "ms"

        if metrics is None:
            metrics = ["RMSE"]

        if "RMSE" not in metrics:
            if "smape_imputer" in metrics:
                to_call = ["smape_imputer", "RUNTIME"]
            elif "f1_imputer" in metrics:
                to_call = ["f1_imputer", "RUNTIME"]
            else:
                to_call = [metrics[0], "RUNTIME"]
        else:
            to_call = ["RMSE", "RUNTIME"]

        new_metrics = np.copy(metrics)

        if metrics is None:
            new_metrics = utils.list_of_metrics()
        else:
            if "RUNTIME" not in new_metrics:
                new_metrics = np.append(new_metrics, "RUNTIME")
            if "RUNTIME_LOG" not in new_metrics:
                new_metrics = np.append(new_metrics, "RUNTIME_LOG")

        if upstream:
            opt = None
            for dataset, patterns_items in runs_plots_scores.items():
                for pattern, algorithm_items in patterns_items.items():
                    for algorithm, optimizer_items in algorithm_items.items():
                        for optimizer, x_data_items in optimizer_items.items():
                            opt = optimizer
                            break

            list_of_patterns = []
            for dataset, patterns_items in runs_plots_scores.items():
                for pattern, algorithm_items in patterns_items.items():
                    list_of_patterns.append(pattern)
                    new_dir = save_dir + "/" + pattern.lower() + "/error"
                    os.makedirs(new_dir, exist_ok=True)

                    save_path = os.path.join(new_dir, f"report_{pattern}_{dataset}.txt")
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    with open(save_path, "w") as file:
                        file.write(f"Report for Dataset: {dataset}\n")
                        file.write(f"Generated on: {current_time}\n")
                        file.write(f"Total runtime: {rt} (ms)\n")
                        file.write(f"Run number: {run}\n")
                        file.write("=" * 120 + "\n\n")

                        for metric in new_metrics:

                            if metric == "RUNTIME":
                                file.write(f"\n{dataset}: {{{pattern}, {metric}[{metric_unit}], {opt}}}")
                            else:
                                file.write(f"\n{dataset}: {{{pattern}, {metric}, {opt}}}")

                            # Collect all algorithms and scores by rate
                            rate_to_scores = defaultdict(dict)
                            all_algorithms = set()

                            for algorithm, optimizer_items in algorithm_items.items():
                                for optimizer, x_data_items in optimizer_items.items():
                                    for x, values in x_data_items.items():
                                        score = values.get("scores", {}).get(metric, None)
                                        if score is not None:
                                            rate_to_scores[x][algorithm] = f"{score:.10f}"
                                            all_algorithms.add(algorithm)

                            all_algorithms = sorted(all_algorithms)
                            headers = ["Rate"] + list(all_algorithms)
                            column_widths = [5] + [18] * len(all_algorithms)

                            # Header and separator rows
                            header_row = "".join(
                                f" {header:^{width}} " for header, width in zip(headers, column_widths))
                            separator_row = "" + "".join(f"{'' * (width + 2)}" for width in column_widths) + ""

                            file.write(f"{separator_row}\n")
                            file.write(f"{header_row}\n")
                            file.write(f"{separator_row}\n")

                            if metric in to_call and verbose:
                                if metric == "RUNTIME":
                                    print(f"\n{dataset}: {{{pattern}, {metric}[{metric_unit}], {opt}}}")
                                else:
                                    print(f"\n{dataset}: {{{pattern}, {metric}, {opt}}}")
                                print(separator_row)
                                print(f"{header_row}")
                                print(separator_row)

                            # Write each row
                            for rate in sorted(rate_to_scores.keys()):
                                row_values = [rate] + [rate_to_scores[rate].get(algo, "") for algo in all_algorithms]
                                row = "".join(f" {val:^{width}} " for val, width in zip(row_values, column_widths))
                                file.write(f"{row}\n")
                                if metric in to_call and verbose:
                                    print(f"{row}")

                            file.write(f"{separator_row}\n\n")
                            if metric in to_call and verbose:
                                print(separator_row + "\n")

                        file.write("Dictionary of Results:\n")
                        file.write(str(runs_plots_scores) + "\n")

        else:
            list_of_patterns = []
            for dataset, patterns_items in runs_plots_scores.items():
                for pattern, algorithm_items in patterns_items.items():
                    for algorithm, models_items in algorithm_items.items():
                        list_of_patterns.append(pattern)
                        new_dir = save_dir + "/" + pattern.lower() + "/error"
                        os.makedirs(new_dir, exist_ok=True)

                        save_path = os.path.join(new_dir, f"report_{pattern}_{dataset.lower()}.txt")
                        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        with open(save_path, "w") as file:
                            file.write(f"Report for Dataset: {dataset}\n")
                            file.write(f"Generated on: {current_time}\n")
                            file.write(f"Total runtime: {rt} (ms)\n")
                            file.write(f"Run number: {run}\n")
                            file.write("=" * 120 + "\n\n")

                            for metric in new_metrics:

                                if metric == "RUNTIME":
                                    file.write(f"\n{dataset}: {{{pattern}, {metric}[{metric_unit}], {algorithm}}}")
                                else:
                                    file.write(f"\n{dataset}: {{{pattern}, {metric}, {algorithm}}}")

                                # Collect all algorithms and scores by rate
                                rate_to_scores = defaultdict(dict)
                                all_models = set()

                                for model, x_data_items in models_items.items():
                                    for x, values in x_data_items.items():
                                        score = values.get("scores", {}).get(metric, None)
                                        if score is not None:
                                            rate_to_scores[x][model] = f"{score:.10f}"
                                            all_models.add(model)

                                all_models = sorted(all_models)
                                headers = ["Rate"] + list(all_models)
                                column_widths = [5] + [18] * len(all_models)

                                # Header and separator rows
                                header_row = "".join(
                                    f" {header:^{width}} " for header, width in zip(headers, column_widths))
                                separator_row = "" + "".join(f"{'' * (width + 2)}" for width in column_widths) + ""

                                file.write(f"{separator_row}\n")
                                file.write(f"{header_row}\n")
                                file.write(f"{separator_row}\n")

                                if metric in to_call and verbose:
                                    if metric == "RUNTIME":
                                        print(f"\n{dataset}: {{{pattern}, {metric}[{metric_unit}], {algorithm}}}")
                                    else:
                                        print(f"\n{dataset}: {{{pattern}, {metric}, {algorithm}}}")
                                    print(separator_row)
                                    print(f"{header_row}")
                                    print(separator_row)

                                # Write each row
                                for rate in sorted(rate_to_scores.keys()):
                                    row_values = [rate] + [rate_to_scores[rate].get(model, "") for model in all_models]
                                    row = "".join(f" {val:^{width}} " for val, width in zip(row_values, column_widths))
                                    file.write(f"{row}\n")
                                    if metric in to_call and verbose:
                                        print(f"{row}")

                                file.write(f"{separator_row}\n\n")
                                if metric in to_call and verbose:
                                    print(separator_row + "\n")

                            file.write("Dictionary of Results:\n")
                            file.write(str(runs_plots_scores) + "\n")

    def generate_plots(self, runs_plots_scores, ticks, metrics=None, subplot=False, y_size=8, title=None, save_dir="./reports", display=False, upstream=True, verbose=True):
        """
        Generate and save plots for each metric and pattern based on provided scores.

        Parameters
        ----------
        runs_plots_scores : dict
            Dictionary containing scores and timing information for each dataset, pattern, and algorithm.
        ticks : list of float
            List of missing rates for contamination.
        metrics : list of string
            List of metrics used
        subplot : bool, optional
            If True, generates a single figure with subplots for all metrics (default is False).
        y_size : int, optional
            Default size of the graph (default is 4).
        title : str, optional
            Title of the graph (default is "imputegap benchmark").
        save_dir : str, optional
            Directory to save generated plots (default is "./reports").
        display : bool, optional
            Display or not the plots (default is False).
        verbose : bool, optional
            Whether to display the contamination information (default is True).

        Returns
        -------
        None

        Notes
        -----
        Saves generated plots in `save_dir`, categorized by dataset, pattern, and metric.
        """
        os.makedirs(save_dir, exist_ok=True)

        markers = itertools.cycle(["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "h", "p", "8"])
        marker_by_algo = {}

        print("\nThe plots have been generated...\n")

        new_metrics = np.copy(metrics)

        new_plots = 0

        if metrics is None:
            new_metrics = utils.list_of_metrics()
        else:
            if "RUNTIME_LOG" not in new_metrics:
                new_plots = new_plots + 1
                new_metrics = np.append(new_metrics, "RUNTIME_LOG")

        nbr_metrics = len(new_metrics)
        n_rows = int((len(new_metrics) + new_plots) / 2)

        x_size, title_flag = 16, title

        if ticks and len(ticks) > 0:
            tick_min = float(min(ticks))
            tick_max = float(max(ticks))
        else:
            tick_min, tick_max = 0.0, 1.0  # fallback

        x_pad = 0.025  # 5% points (because rates are in [0,1])
        x_left = max(0.0, tick_min - x_pad)
        x_right = min(1.0, tick_max + x_pad)

        for dataset, pattern_items in runs_plots_scores.items():
            for pattern, algo_items in pattern_items.items():
                if subplot:
                    x_size = x_size * 2
                    y_size = y_size * round(nbr_metrics // 2)
                    scale_factor = 0.85
                    x_size_screen = (1920 / 100) * scale_factor
                    y_size_screen = (1080 / 100) * scale_factor
                    if n_rows < 4:
                        x_size = x_size_screen
                        y_size = y_size_screen

                    ncols = 2
                    if nbr_metrics % 2 == 1:
                        ncols, n_rows, y_size = 1, (n_rows * 2) - 1, y_size * 1.25

                    fig, axes = plt.subplots(nrows=n_rows, ncols=ncols, figsize=(x_size, y_size))  # Adjusted figsize
                    axes = axes.ravel()

                    fig.subplots_adjust(
                        left=0.04,
                        right=0.99,
                        top=0.97,
                        bottom=0.05,
                        wspace=0.095,
                        hspace=0.35
                    )

                    if title_flag is None:
                        title = dataset + " : " + pattern + ", benchmark analysis"
                    fig.canvas.manager.set_window_title(title)

                # Iterate over each metric, generating separate plots, including new timing metrics
                for i, metric in enumerate(new_metrics):
                    if subplot:
                        if i < len(axes):
                            ax = axes[i]
                        else:
                            break  # Prevent index out of bounds if metrics exceed subplot slots
                    else:
                        plt.figure(figsize=(x_size, y_size))
                        ax = plt.gca()

                    has_data = False  # Flag to check if any data is added to the plot
                    max_y, min_y = -99999, 99999

                    if upstream:
                        for algorithm, optimizer_items in algo_items.items():
                            x_vals = []
                            y_vals = []
                            for optimizer, x_data in optimizer_items.items():
                                for x, values in x_data.items():
                                    if metric in values["scores"]:
                                        x_vals.append(float(x))
                                        y_vals.append(values["scores"][metric])

                            if x_vals and y_vals:
                                sorted_pairs = sorted(zip(x_vals, y_vals))
                                x_vals, y_vals = zip(*sorted_pairs)

                                if algorithm not in marker_by_algo:
                                    marker_by_algo[algorithm] = next(markers)
                                m = marker_by_algo[algorithm]

                                # Plot each algorithm as a line with scattered points
                                ax.plot(x_vals, y_vals, label=f"{algorithm}", linewidth=2, marker=m, markersize=6)
                                ax.scatter(x_vals, y_vals, marker=m, s=35)

                                # ax.plot(x_vals, y_vals, label=f"{algorithm}", linewidth=2)
                                # ax.scatter(x_vals, y_vals)
                                has_data = True

                                if min_y > min(y_vals):
                                    min_y = min(y_vals)
                                if max_y < max(y_vals):
                                    max_y = max(y_vals)

                            file_token = "by_algorithm"

                    else:
                        # model_points[optimizer][rate] -> list of y-values (one per algorithm)
                        model_points = defaultdict(lambda: defaultdict(list))

                        for algorithm, models_items in algo_items.items():
                            for model, x_data in models_items.items():
                                for x, values in x_data.items():
                                    if metric in values.get("scores", {}):
                                        model_points[model][float(x)].append(values["scores"][metric])

                        for model, rate_map in model_points.items():
                            x_vals, y_vals = [], []
                            for xf in sorted(rate_map.keys()):
                                vals = rate_map[xf]
                                if not vals:
                                    continue
                                # aggregate across algorithms
                                y = float(np.mean(vals))
                                x_vals.append(xf)
                                y_vals.append(y)

                            if x_vals and y_vals:
                                if model not in marker_by_algo:
                                    marker_by_algo[model] = next(markers)
                                m = marker_by_algo[model]

                                ax.plot(x_vals, y_vals, label=f"{model}", linewidth=2, marker=m, markersize=6)
                                ax.scatter(x_vals, y_vals, marker=m, s=35)

                                has_data = True
                                min_y = min(min_y, min(y_vals))
                                max_y = max(max_y, max(y_vals))

                        file_token = "by_model"

                    # Save plot only if there is data to display
                    if has_data:
                        ylabel_metric = {
                            "RUNTIME": "Runtime [ms]",
                            "RUNTIME_LOG": "log₁₀(Runtime [ms])",
                        }.get(metric, metric)

                        ax.set_title(metric)
                        ax.set_xlabel("Rate")
                        ax.set_ylabel(ylabel_metric)
                        # ax.set_xlim(0.0, 0.85)
                        ax.set_xlim(x_left, x_right)

                        bounds = {"RMSE": (0, 3), "MAE": (0, 3), "CORRELATION": (-1, 1), "MI": (0, 2),
                                  "RUNTIME": (0, 10000), "RUNTIME_LOG": (-5, 5), }

                        if metric in bounds:
                            lo, hi = bounds[metric]
                            min_y = max(min_y, lo)
                            max_y = min(max_y, hi)

                        diff = (max_y - min_y)
                        y_padding = 0.15 * diff

                        if y_padding is None or y_padding == 0:
                            y_padding = 1

                        ax.set_ylim(min_y - y_padding, max_y + y_padding)

                        # Set y-axis limits with padding below 0 for visibility
                        if metric == "RUNTIME":
                            ax.set_title("Runtime (linear scale)")
                        elif metric == "RUNTIME_LOG":
                            ax.set_title("Runtime (log scale)")
                        elif metric == "CORRELATION":
                            ax.set_title("Pearson Correlation")

                        # Customize x-axis ticks
                        ax.set_xticks(ticks)
                        ax.set_xticklabels([f"{int(tick * 100)}%" for tick in ticks])
                        ax.grid(True, zorder=0)
                        ax.legend(loc='upper left', fontsize=7, frameon=True, fancybox=True, framealpha=0.8,
                                  ncol=len(ax.get_legend_handles_labels()[0]))

                    if not subplot:
                        new_dir = save_dir + "/" + pattern
                        os.makedirs(new_dir, exist_ok=True)
                        filepath = os.path.join(new_dir, f"{dataset}_{pattern}_{file_token}_{metric}.jpg")
                        plt.savefig(filepath)
                        if not display:
                            plt.close()

                if subplot:
                    # plt.tight_layout()
                    new_dir = save_dir + "/" + pattern + "/error"
                    os.makedirs(new_dir, exist_ok=True)
                    filename = f"{dataset}_{pattern}_metrics_subplot.jpg"
                    filepath = os.path.join(new_dir, filename)
                    plt.savefig(filepath)

        self.subplots = plt

    def generate_avg_bar_plot(self, results, save_path, type=None, upstream=False, plots=True, verbose=True):
        """
        Generate summary statistics and per-metric boxplots aggregated across benchmark runs.

        The downstream metric used depends on the `type` argument:

        - type="classifier"  -> uses "DOWNSTREAM_ACC"
        - type="forecaster"  -> uses "DOWNSTREAM_SMAPE"
        - otherwise          -> uses "DOWNSTREAM"


        Parameters
        ----------
        results : dict | list[dict] | str
            Either a nested results dictionary (or list of such dicts), or a string path to a log file containing
            a dumped results object. Expected nested structure: results[dataset][pattern][algorithm][model][rate]["scores"][metric] = value

        save_path : str
            Output directory where the summary log and plots will be saved.

        type : str or None, optional
            Determines which downstream metric key to include and which y-limits to apply:

            - "classifier": downstream key = "DOWNSTREAM_ACC"
            - "forecaster": downstream key = "DOWNSTREAM_SMAPE"
            - None/other : downstream key = "DOWNSTREAM"

            Default is None.

        verbose : bool, optional
            If True, prints summary lines to the console. Default is True.

        Returns
        -------
        None
            This function produces side effects (writes files and generates plots).

        """

        if isinstance(results, str):

            text = Path(results).read_text(errors="ignore")

            marker = "Dictionary of Results:"
            pos = text.rfind(marker)
            if pos == -1:
                raise ValueError("Could not find 'Dictionary of Results:' in the log.")

            tail = text[pos + len(marker):].lstrip()
            m = re.search(r"[\[{]", tail)
            if not m:
                raise ValueError("Could not find start of dumped object ('[' or '{').")

            s = tail[m.start():]
            open_ch = s[0]
            close_ch = "]" if open_ch == "[" else "}"

            depth = 0
            end = None
            for i, ch in enumerate(s):
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end is None:
                raise ValueError("Unbalanced brackets while extracting dumped object.")

            expr = s[:end]
            expr = re.sub(r"\bnp\.float(16|32|64)\(", "float(", expr)
            expr = re.sub(r"\bnan\b", "float('nan')", expr)

            results_obj = eval(expr, {"__builtins__": {}}, {"float": float})
            dicts = [results_obj] if isinstance(results_obj, dict) else results_obj

        else:
            dicts = results
        # -----------------------------
        # Flatten to rows for stats/boxplot
        # -----------------------------
        rows = []
        for d in dicts:
            for dataset, patterns in d.items():
                for pattern, algos in patterns.items():
                    for algorithm, models in algos.items():
                        for model, rates in models.items():
                            for rate, payload in rates.items():
                                scores = payload.get("scores", {})
                                for metric, val in scores.items():
                                    try:
                                        v = float(val)
                                    except Exception:
                                        v = float("nan")
                                    rows.append(
                                        {
                                            "dataset": str(dataset),
                                            "pattern": str(pattern),
                                            "algorithm": str(algorithm),
                                            "model": str(model),
                                            "rate": str(rate),
                                            "metric": str(metric),
                                            "value": v,
                                        }
                                    )

        df = pd.DataFrame(rows)

        if type == "classifier":
            down = "DOWNSTREAM_ACC"
            YLIMS = {"RMSE": (0, 1.0), "DOWNSTREAM": (0, 1), "RUNTIME": (0, 5), }
        elif type == "classifier-down":
            down = "DOWNSTREAM"
            YLIMS = {"accuracy_groundtruth": (0, 1), "accuracy_imputer": (0, 1), "accuracy_meanimpute": (0, 1),
                     "recall_groundtruth": (0, 1), "recall_imputer": (0, 1), "recall_meanimpute": (0, 1),
                     "f1_groundtruth": (0, 1), "f1_imputer": (0, 1), "f1_meanimpute": (0, 1), "RMSE": (0, 1.0),
                     "RUNTIME": (0, 5), }
        elif type == "classifier-forecaster":
            down = "DOWNSTREAM"
            YLIMS = {"mse_groundtruth": (0, 1.0), "mse_imputer": (0, 1.0), "mse_baseline": (0, 1.0),
                     "mae_groundtruth": (0, 1.0), "mae_imputer": (0, 1.0), "mae_baseline": (0, 1.0),
                     "smape_groundtruth": (0, 1.6), "smape_imputer": (0, 1.6), "smape_baseline": (0, 1.6), }
        elif type == "forecaster":
            down = "DOWNSTREAM_SMAPE"
            YLIMS = {"RMSE": (0, 1.0), "DOWNSTREAM": (0, 2.5), "RUNTIME": (0, 5), }
        else:
            down = "DOWNSTREAM"
            type = "downstream"
            YLIMS = {"RMSE": (0, 1.0), "DOWNSTREAM": (0, 1), "RUNTIME": (0, 5), }

        # -----------------------------
        # Stats per algorithm (across all dataset/pattern/model/rate)
        # mean, std, median, q1, q3, min, max, count
        # -----------------------------
        if type == "classifier-down":
            METRICS_TO_PRINT = ["accuracy_groundtruth", "accuracy_imputer", "accuracy_meanimpute", "recall_groundtruth",
                                "recall_imputer", "recall_meanimpute", "f1_groundtruth", "f1_imputer", "f1_meanimpute",
                                "RMSE", "RUNTIME"]
        elif type == "classifier-forecaster":
            METRICS_TO_PRINT = ["mse_groundtruth", "mse_imputer", "mse_baseline", "mae_groundtruth", "mae_imputer",
                                "mae_baseline", "smape_groundtruth", "smape_imputer", "smape_baseline", "RMSE",
                                "RUNTIME"]
        else:
            METRICS_TO_PRINT = ["RMSE", down, "RUNTIME"]  # edit as you want

        df_sub = df[df["metric"].isin(METRICS_TO_PRINT)].copy()

        group_col = "algorithm" if upstream else "model"

        stats = (
            df_sub.groupby([group_col, "metric"])["value"]
            .agg(
                count="count",
                mean="mean",
                std="std",
                median="median",
                q1=lambda s: s.quantile(0.25),
                q3=lambda s: s.quantile(0.75),
                min="min",
                max="max",
            )
            .reset_index()
        )

        stats.to_csv(os.path.join(save_path, "_stats_results.csv"), index=False)

        lines = []
        lines.append(f"--- Summary (mean ± std) per {group_col} ---\n")

        # Precompute MeanImpute RMSE mean if available
        meanimpute_rmse = None

        if group_col == "algorithm":
            sub_mi = stats[(stats["algorithm"] == "MeanImpute") & (stats["metric"] == "RMSE")]
            if not sub_mi.empty and not pd.isna(sub_mi["mean"].iloc[0]):
                meanimpute_rmse = float(sub_mi["mean"].iloc[0])

        # Print like: alg = RMSE mean±std, ...
        if verbose:
            print("\n--- Summary (mean ± std) per algorithm ---\n")

        for alg in sorted(stats[group_col].unique()):
            parts = []

            rmse_mean = None
            for metric in METRICS_TO_PRINT:
                sub = stats[(stats[group_col] == alg) & (stats["metric"] == metric)]
                if sub.empty:
                    continue
                mean_v = float(sub["mean"].iloc[0])
                std_v = float(sub["std"].iloc[0]) if not pd.isna(sub["std"].iloc[0]) else float("nan")
                parts.append(f"{metric}={mean_v:.6g}±{std_v:.6g}")
                if metric == "RMSE":
                    rmse_mean = mean_v

            if (meanimpute_rmse is not None and rmse_mean is not None and alg != "MeanImpute"):
                delta = meanimpute_rmse - rmse_mean
                parts.append(f"DELTA={delta:.6g}")
            if parts:
                if verbose:
                    print(f"{alg} = " + ", ".join(parts), "\n")
                lines.append(f"{alg} = " + ", ".join(parts) + "\n")

        os.makedirs(save_path, exist_ok=True)

        report = os.path.join(save_path, "average_results.log")
        with open(report, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        # -----------------------------
        # Box plot prep (one figure per metric)
        # -----------------------------
        for metric in METRICS_TO_PRINT:
            dfx = df_sub[df_sub["metric"] == metric].dropna(subset=["value"])
            if dfx.empty:
                continue

            algs = sorted(dfx[group_col].unique())
            data = [dfx[dfx[group_col] == a]["value"].values for a in algs]

            plt.figure()
            plt.boxplot(data, tick_labels=algs, showfliers=False)
            if metric in YLIMS:
                plt.ylim(*YLIMS[metric])  # <-- your requested limits

            plt.title(f"Box plot of {metric} by algorithm")
            plt.ylabel(metric)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            filename = f"{type}_average_{metric}.jpg"
            dir = os.path.join(save_path, "_barplot")
            os.makedirs(dir, exist_ok=True)
            filepath = os.path.join(dir, filename)
            plt.savefig(filepath)

            if plots:
                if (metric == "RMSE" and upstream) or (metric == "f1_imputer" and not upstream) or (
                        metric == "smape_imputer" and not upstream):
                    plt.show()
                else:
                    plt.close('all')
            else:
                plt.close('all')

    def eval(self, algorithms=None, datasets=None, patterns=None, x_axis=None, optimizer="default_params", metrics=None, save_dir="./imputegap_assets/benchmark", runs=1, normalizer="z_score", report_title="", nbr_series=200, nbr_vals=2000, dl_ratio=None, verbose=False):
        """
        Execute a comprehensive evaluation of imputation algorithms over multiple datasets and patterns.

        Parameters
        ----------
        algorithms : list of str
            List of imputation algorithms to test. If None, default is ["cdrec"]

        datasets : list of str
            List of dataset names to evaluate. If None, default is ["eeg-alcohol"]

        patterns : list of str
            List of contamination patterns to apply. If None, default is ["mcar"]

        x_axis : list of float
            List of missing rates for contamination. If None, default is [0.1, 0.2, 0.4, 0.6, 0.8]

        optimizer : str, dict
            Name of the optimizer (str) or optimizer with their configurations (dict).

        metrics : list of str
            List of metrics for evaluation. If None, default is ["*"]

        save_dir : str, optional
            Directory to save reports and plots (default is "./reports").

        runs : int, optional
            Number of executions with a view to averaging them

        normalizer : str, optional
            Normalizer to pre-process the data (default is "z_score").

        report_title : str, optional
            Title of the report (default is "").

        nbr_series : int, optional
            Number of series to take inside the dataset (default is 200 (as the max values)). Set to None to remove the limitation.

        nbr_vals : int, optional
            Number of values to take inside the series (default is 2500 (as the max values)). Set to None to remove the limitation.

        dl_ratio : float, optional
            Training ratio for Deep Learning techniques (default is 0.8)

        verbose : bool, optional
            Whether to display the contamination information (default is False).

        Returns
        -------
        List
            List of all runs results, matrix with averaged scores and times for all levels

        Notes
        -----
        Runs contamination, imputation, and evaluation, then generates plots and a summary reports.
        """

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        run_storage = []
        not_optimized = ["none"]
        mean_group = ["mean", "MeanImpute", "min", "MinImpute", "zero", "ZeroImpute", "MeanImputeBySeries",
                      "meanimpute", "minimpute", "zeroimpute", "meanimputebyseries"]

        if algorithms is None:
            algorithms = ["cdrec"]
        if datasets is None:
            datasets = ["eeg-alcohol"]
        if patterns is None:
            patterns = ["mcar"]
        if x_axis is None:
            x_axis = [0.1, 0.2, 0.4, 0.6, 0.8]
        if metrics is None:
            metrics = ["*"]

        if optimizer is None:
            optimizer = "default_params"

        if not isinstance(algorithms, list):
            raise TypeError(f"'algorithms' must be a list, but got {type(algorithms).__name__}")
        if not isinstance(datasets, list):
            raise TypeError(f"'datasets' must be a list, but got {type(datasets).__name__}")
        if not isinstance(patterns, list):
            raise TypeError(f"'patterns' must be a list, but got {type(patterns).__name__}")
        if not isinstance(x_axis, list):
            raise TypeError(f"'x_axis' must be a list, but got {type(x_axis).__name__}")
        if not isinstance(optimizer, str) and not isinstance(optimizer, dict):
            raise TypeError(f"'optimizer' must be a str or dict, but got {type(optimizer).__name__}")

        if "*" in metrics or "all" in metrics:
            metrics = utils.list_of_metrics()
        if "*" in metrics or "all" in algorithms:
            algorithms = utils.list_of_algorithms()

        if "default" in optimizer and isinstance(optimizer, str):
            optimizer = "default_params"

        directory_now = datetime.datetime.now()
        directory_time = directory_now.strftime("%y_%m_%d_%H_%M_%S")
        save_dir = save_dir + "/" + "benchmark_" + report_title + "_" + directory_time

        if nbr_series is None:
            nbr_series = 10000000
        if nbr_vals is None:
            nbr_vals = 10000000

        benchmark_time = time.time()

        definition_of_exp = f"\nThe benchmark has been called:\n\talgorithms: {algorithms}\n\tdatasets: {datasets}\n\tpatterns: {patterns}\n\tmissing_percentages: {x_axis}\n\toptimizer: {optimizer}\n\tnormalizer: {normalizer}\n\truns: {runs}\n\tnumber max series: {nbr_series}\n\tnumber max values: {nbr_vals}\n\n"
        print(definition_of_exp)

        for i_run in range(0, abs(runs)):
            for dataset in datasets:
                runs_plots_scores = {}
                block_size_mcar = 10
                y_p_size = max(4, len(algorithms) * 0.275)

                if verbose:
                    print("\n1. evaluation launch for", dataset, "\n")
                ts_test = TimeSeries(verbose=False)
                default_data = TimeSeries(verbose=False)

                header = False
                if dataset == "eeg-reading" or dataset == "eegreading":
                    header = True

                reshp = False
                default_data.load_series(data=utils.search_path(dataset), header=header, verbose=False)
                Ndef, Mdef = default_data.data.shape

                if Ndef > nbr_vals or Mdef > nbr_series:
                    reshp = True
                    print(
                        f"\nThe dataset {dataset} contains a large number of values {default_data.data.shape}, which may be too much for some algorithms to handle efficiently. Consider reducing the number of series or the volume of data.")
                default_data = None

                ts_test.load_series(data=utils.search_path(dataset), nbr_series=nbr_series, nbr_val=nbr_vals,
                                    header=header, normalizer=normalizer, verbose=verbose)
                N, M = ts_test.data.shape

                if M <= 0:
                    raise ValueError(f"The dataset loaded has no series (series {M}).")

                if reshp:
                    print(f"Benchmarking module has reduced the shape to {ts_test.data.shape}.\n")

                if N < 250:
                    block_size_mcar = 2
                    print(f"The block size is too high for the number of values per series, reduce to {block_size_mcar}\n")

                for pattern in patterns:
                    if verbose:
                        print("\n2. contamination of", dataset, "with pattern", pattern, "\n")

                    for algorithm in algorithms:
                        has_been_optimized = False

                        if verbose:
                            print(
                                f"3. {algorithm} is tested with {pattern} on {dataset}, started at {time.strftime('%Y-%m-%d %H:%M:%S')}.")
                        else:
                            print(
                                f"{algorithm} is tested with {pattern} on {dataset}, started at {time.strftime('%Y-%m-%d %H:%M:%S')}.")

                        for incx, x in enumerate(x_axis):
                            if verbose:
                                print("\n4. missing values (series&values) set to", x, "for x_axis\n")

                            incomp_data = utils.config_contamination(ts=ts_test, pattern=pattern, dataset_rate=x,
                                                                     series_rate=x, block_size=block_size_mcar,
                                                                     verbose=verbose)

                            opt_imp = optimizer

                            try:
                                algo = utils.config_impute_algorithm(incomp_data=incomp_data, algorithm=algorithm,
                                                                     verbose=verbose)

                                if not isinstance(opt_imp, dict) and opt_imp != "default_params":
                                    if opt_imp == "ray-tune":
                                        opt_imp = "ray_tune"
                                    opt_imp = {"optimizer": opt_imp}

                                if isinstance(opt_imp, dict):
                                    optimizer_gt = {"input_data": ts_test.data, **opt_imp}
                                    optimizer_value = opt_imp.get('optimizer')  # or optimizer['optimizer']
                                    if not has_been_optimized and algorithm not in mean_group and algorithm not in not_optimized:

                                        if verbose:
                                            print("\n5. AutoML to set the parameters", opt_imp, "\n")
                                        i_opti = self._config_optimization(0.20, ts_test, pattern, algorithm,
                                                                           block_size_mcar)

                                        if utils.check_family("DeepLearning", algorithm):
                                            if dl_ratio is None:
                                                i_opti.impute(user_def=False, params=optimizer_gt)
                                            else:
                                                i_opti.impute(user_def=False, params=optimizer_gt, tr_ratio=dl_ratio)
                                        else:
                                            i_opti.impute(user_def=False, params=optimizer_gt)

                                        optimal_params_path = utils.save_optimization(optimal_params=i_opti.parameters,
                                                                                      algorithm=algorithm,
                                                                                      dataset=dataset, optimizer="e",
                                                                                      verbose=verbose)

                                        has_been_optimized = True
                                    else:
                                        if verbose:
                                            print("\n5. AutoML already optimized...\n")

                                    if algorithm not in mean_group and algorithm not in not_optimized:
                                        if i_opti.parameters is None:
                                            opti_params = utils.load_parameters(query="optimal", algorithm=algorithm,
                                                                                dataset=dataset, optimizer="e",
                                                                                path=optimal_params_path,
                                                                                verbose=verbose)
                                            if verbose:
                                                print("\n6. load imputation", algorithm,
                                                      "with optimal parameters from files", *opti_params)
                                        else:
                                            opti_params = i_opti.parameters
                                            if verbose:
                                                print("\n6. set imputation", algorithm,
                                                      "with optimal parameters from object", *opti_params)
                                    else:
                                        if verbose:
                                            print("\n5. No AutoML launches without optimal params for", algorithm, "\n")
                                        opti_params = None
                                else:
                                    if verbose:
                                        print("\n5. Default parameters have been set the parameters", opt_imp, "for",
                                              algorithm, "\n")
                                    optimizer_value = opt_imp
                                    opti_params = None

                                start_time_imputation = time.time()

                                #if not self._benchmark_exception(dataset, algorithm, pattern, x, N, M):
                                if (utils.check_family("DeepLearning", algorithm) or utils.check_family("LLMs", algorithm)) and dl_ratio is not None:
                                    if x > round(1 - dl_ratio, 2):
                                        algo.recov_data = incomp_data
                                    else:
                                        algo.impute(params=opti_params, tr_ratio=dl_ratio)
                                else:
                                    algo.impute(params=opti_params)
                                #else:
                                #algo.recov_data = incomp_data

                                end_time_imputation = time.time()

                                algo.score(input_data=ts_test.data, recov_data=algo.recov_data, verbose=False)

                                if "*" not in metrics and "all" not in metrics:
                                    algo.metrics = {k: algo.metrics[k] for k in metrics if k in algo.metrics}

                                time_imputation = (end_time_imputation - start_time_imputation) * 1000
                                time_imputation = max(time_imputation, 1)
                                log_time_imputation = math.log10(time_imputation) if time_imputation > 0 else None

                                algo.metrics["RUNTIME"] = time_imputation
                                algo.metrics["RUNTIME_LOG"] = log_time_imputation

                                dataset_s = dataset
                                if "-" in dataset:
                                    dataset_s = dataset.replace("-", "")

                                save_dir_plot = save_dir + "/" + dataset_s + "/" + pattern + "/recovery/"
                                cont_rate = int(x * 100)
                                ts_test.plot(input_data=ts_test.data, incomp_data=incomp_data,
                                             recov_data=algo.recov_data, nbr_series=6, subplot=True,
                                             algorithm=algo.algorithm, cont_rate=str(cont_rate), display=False,
                                             save_path=save_dir_plot, verbose=False)

                                runs_plots_scores.setdefault(str(dataset_s), {}).setdefault(str(pattern),
                                                                                            {}).setdefault(
                                    str(algorithm), {}).setdefault(str(optimizer_value), {})[str(x)] = {
                                    "scores": algo.metrics}

                            except Exception as e:
                                dataset_s = dataset
                                if "-" in dataset:
                                    dataset_s = dataset.replace("-", "")

                                print(f"Error during benchmark for {algorithm}, with {dataset_s}, and {x}%: {e}")

                                algo.metrics = {
                                    "RMSE": np.nan,
                                    "MAE": np.nan,
                                    "MI": np.nan,
                                    "CORRELATION": np.nan,
                                    "RUNTIME": np.nan,
                                    "RUNTIME_LOG": np.nan,
                                }

                                if isinstance(opt_imp, dict):
                                    val_opt = opt_imp.get("optimizer")
                                if isinstance(opt_imp, str):
                                    val_opt = opt_imp
                                if val_opt is None:
                                    val_opt = ""

                                runs_plots_scores.setdefault(str(dataset_s), {}).setdefault(str(pattern),
                                                                                            {}).setdefault(
                                    str(algorithm), {}).setdefault(str(val_opt), {})[str(x)] = {"scores": algo.metrics}

                                os.makedirs(save_dir, exist_ok=True)
                                save_path = os.path.join(save_dir, "error.log")
                                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                with open(save_path, "a") as file:
                                    file.write(
                                        f"{timestamp} | Error during benchmark for {algorithm}, with {dataset_s} - for a shape of ({N}, {M}) - ({pattern}/{val_opt}), and {x}%: {e}\n\n")

                        print(f"done!\n\n")

                run_storage.append(runs_plots_scores)

        plt.close('all')  # Close all open figures

        for x, m in enumerate(reversed(metrics)):
            # tag = True if x == (len(metrics)-1) else False
            scores_list, algos, sets = self.avg_results_task(*run_storage, metric=m)
            _ = self.generate_heatmap(scores_list=scores_list, algos=algos, sets=sets, metric=m, save_dir=save_dir,
                                      display=False)

        run_averaged = self.average_runs_by_names(run_storage)

        benchmark_end = time.time()
        total_time_benchmark = round(benchmark_end - benchmark_time, 4)
        print(f"\n> logs: benchmark - Execution Time: {total_time_benchmark} seconds\n")

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "runtime.log")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(save_path, "a") as file:
            file.write(f"{timestamp} | logs: benchmark - Execution Time: {total_time_benchmark} seconds\n")

        verb = False

        for scores in run_averaged:
            all_keys = list(scores.keys())
            dataset_name = str(all_keys[0])
            save_dir_agg_set = save_dir + "/" + dataset_name

            self.generate_reports_txt(runs_plots_scores=scores, save_dir=save_dir_agg_set, given_dataset=dataset_name, metrics=metrics, rt=total_time_benchmark, run=-1)
            self.generate_plots(runs_plots_scores=scores, ticks=x_axis, metrics=metrics, subplot=True, y_size=y_p_size, save_dir=save_dir_agg_set, display=verb)

        self.generate_reports_summary(run_of_values=run_averaged, save_dir=save_dir, metrics=metrics,
                                      rt=total_time_benchmark, run=-1, title=report_title)

        print("\nThe results are saved in : ", save_dir, "\n")
        self.list_results = run_averaged
        self.aggregate_results = scores_list

        save_def = os.path.join(save_dir, "experimentation_setup.log")
        with open(save_def, "w") as file:
            file.write(definition_of_exp)

    def eval_downstream_classification(self, classifiers=None, algorithms=None, datasets=None, patterns=None, x_axis=None, metrics=None, save_dir="./imputegap_assets/benchmark", runs=1, normalizer="z_score", contamination_by_class=True, imputation_by_class=True, report_title="", nbr_series=10000, nbr_vals=1000, test_imputation=False, to_cache=True, use_cache=True, run_downstream=False, fixed_rate=0.2, evaluate_upstream=True, bypass_error=False, generate_plot=True, plots=True, inner_plots=True, referential=True, verbose=False, v2=False):
        """
        Benchmark imputation/recovery algorithms using a downstream classification task across datasets, missingness patterns, and missingness rates.

        This routine runs an end-to-end benchmark loop:

        1) Load each classification dataset (optionally normalized and optionally prepared for test-set imputation).

        2) Generate missingness either globally or class-conditionally (controlled by `contamination_class`) for multiple missingness rates (`x_axis`) and patterns (`patterns`).

        3) Impute/recover the contaminated training data (and optionally test data) using each specified algorithm.

        4) Score the recovered matrix with reconstruction metrics (e.g., RMSE/MAE/etc.) and a downstream classification metric extracted from `imputer_tr.downstream_metrics`

        5) Save plots, logs, and aggregated reports (heatmaps, line plots, summary tables).

        The benchmark results are stored per run in a nested dictionary structure: results[dataset][pattern][algorithm][classifier][missing_rate] = {"scores": imputer_tr.metrics}


        Parameters
        ----------
        classifiers : list[str], optional
            Classification model identifiers to evaluate downstream. All classifiers are stored into the function `utils.list_of_classifiers()`. If None, default is ["arsenal"].

        algorithms : list[str], optional
            Imputation/recovery algorithms to benchmark. If "*" / "all" is used, it is replaced by `utils.list_of_algorithms()`. If None, default is ["cdrec"].

        datasets : list[str], optional
            Classification datasets to benchmark. If None, default is ["Car"]. For all, you can use : utils.get_datasets_classifiers()

        patterns : list[str], optional
            Missingness/contamination patterns (e.g., "mcar", "aligned_series", "aligned_timestamps"). If None, default is ["mcar"].

        x_axis : list[float], optional
            Missingness rates used for contamination (interpreted as proportions; e.g. 0.1 = 10%). If None, default is [0.1, 0.2, 0.4, 0.6, 0.8].

        metrics : list[str], optional
            Subset of reconstruction metrics to keep in `imputer_tr.metrics`. If "*" / "all" is present, uses `utils.list_of_metrics()`. If None, default is ["*"].

        save_dir : str, optional
            Base output directory for saving benchmark artifacts. A timestamped subdirectory is created. Default is "./imputegap_assets/benchmark".

        runs : int, optional
            Number of repeated benchmark runs. The absolute value is used in the loop. Default is 1.

        normalizer : str, optional
            Normalization method used when loading datasets (passed to `ts.load_classify_dataset`). Default is "z_score".

        contamination_by_class : bool, optional
            If True, contamination/missingness is generated in a class-conditional manner (each class can be contaminated independently > advised).
            If False, a single global missingness mask is generated first and then applied across classes.
            Default is True.

        imputation_by_class : bool, optional
            If True, runs the imputation/recovery step separately per class partition (class-wise imputation> advised).
            If False, imputation is performed on the full contaminated training matrix.
            Default is True.

        report_title : str, optional
            Optional title string inserted into the output directory name and summary report. Default is "".

        nbr_series : int or None, optional
            Maximum number of series to include in evaluation (used by dataset loader). If None, treated as very
            large. Default is 10000.

        nbr_vals : int or None, optional
            Maximum number of values/timestamps to include in evaluation (used by dataset loader). If None,
            treated as very large. Default is 1000.

        test_imputation : bool, optional
            Forwarded to `ts.load_classify_dataset(...)` to control whether the dataset preparation includes a
            test-set imputation setup (i.e., generating/keeping test contamination artifacts).
            Default is False.

        to_cache : bool, optional
            If True, stores intermediate benchmark artifacts/results into cache (when supported by your pipeline),
            so repeated runs can be faster/reproducible. Default is True.

        use_cache : bool, optional
            If True, attempts to load cached intermediate artifacts/results instead of recomputing them.
            Default is True.

        run_downstream : bool, optional
            If True, executes downstream classification evaluation and stores downstream metrics (e.g. accuracy)
            alongside reconstruction metrics. If False, skips downstream evaluation.
            Default is False.

        fixed_rate : float, optional
            Fixed missingness rate used in contexts where a single rate is required (e.g., some per-class routines,
            baseline comparisons, or when overriding dynamic rate selection). Default is 0.2.

        evaluate_upstream : bool, optional
            If True, computes and records upstream/reconstruction metrics (e.g., RMSE/MAE/MI/CORRELATION).
            If False, computes and records downstream metrics.

        bypass_error : bool, optional
            If True, this variable will put automatically NaN values if an error is encountered.
            Put to true if you want to get the results of all success imputer

        generate_plot: bool, optional
            If True, generates and saves plots (e.g., heatmaps/line plots) for benchmark results.
            Default is True.

        plots : bool, optional
            If True, display plots (e.g., heatmaps/line plots) for benchmark results.
            Default is True.

        inner_plots : bool, optional
            If True, generates and saves every steps plots for each individual datasets for benchmark results.
            Default is True.

        referential: bool, optional
            If True, run the referential algorithm to compare the benchmark results (downstream)

        verbose : bool, optional
            Verbosity flag controlling detailed logging and plotting behavior. Default is False.

        v2: bool, optional
            Default is False, only activate if you want multiple cached matrices.

        Example
        -------
        .. code-block:: python

            from imputegap.recovery.benchmark import Benchmark
            from imputegap.tools import utils

            bench = Benchmark()
            bench.eval_downstream_classification(classifiers=["arsenal"],
                                                 algorithms=["ZeroImpute", "TRMF"], # for all: utils.list_of_algorithms()
                                                 datasets=["Car", "Computers", "Wine"], # for all: utils.get_datasets_classifiers(),
                                                 patterns=["mcar", "aligned_series", "aligned_timestamps"],
                                                 x_axis=[0.1, 0.2, 0.4, 0.6, 0.8],
                                                 metrics=["RMSE", "DOWNSTREAM_ACC", "MI", "CORRELATION", "RUNTIME"],
                                                 normalizer="z-score",
                                                 report_title="cleanimp_benchmark_cl_up",
                                                 contamination_by_class=True,
                                                 imputation_by_class=True,
                                                 bypass_error=False,
                                                 evaluate_upstream=True,
                                                 to_cache=True,
                                                 use_cache=True,
                                                 run_downstream=False,
                                                 fixed_rate=0.2,
                                                 inner_plots=False,
                                                 referential=False,
                                                 plots=True,
                                                 generate_plot=True,
                                                 verbose=False)
        """
        from imputegap.recovery.downstream import Classifier
        from imputegap.recovery.downstream import Artifact
        import gc
        import torch
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        if classifiers is None:
            classifiers = ["arsenal"]
        if algorithms is None:
            algorithms = ["cdrec"]
        if datasets is None:
            datasets = ["Car"]
        if patterns is None:
            patterns = ["mcar"]
        if x_axis is None:
            x_axis = [0.1, 0.2, 0.4, 0.6, 0.8]
        if metrics is None:
            metrics = ["*"]

        # =============================================================================================================
        # PREPARATION OF THE DATA
        # =============================================================================================================
        run_storage = []
        if not isinstance(algorithms, list):
            raise TypeError(f"'algorithms' must be a list, but got {type(algorithms).__name__}")
        if not isinstance(datasets, list):
            raise TypeError(f"'datasets' must be a list, but got {type(datasets).__name__}")
        if not isinstance(patterns, list):
            raise TypeError(f"'patterns' must be a list, but got {type(patterns).__name__}")
        if not isinstance(x_axis, list):
            raise TypeError(f"'x_axis' must be a list, but got {type(x_axis).__name__}")
        if not isinstance(classifiers, list):
            raise TypeError(f"'classifiers' must be a list, but got {type(classifiers).__name__}")

        if metrics is None or ("*" in metrics or "all" in metrics):
            metrics = utils.list_of_metrics()
        if "*" in algorithms or "all" in algorithms:
            algorithms = utils.list_of_algorithms()
        if "*" in classifiers or "all" in classifiers:
            classifiers = utils.list_of_classifiers()
        if "*" in datasets or "all" in datasets:
            datasets = utils.get_datasets_classifiers()
        if "*" in patterns or "all" in patterns:
            patterns = ["mcar", "aligned_series", "aligned_timestamps"]
        if "*" in x_axis or "all" in x_axis:
            x_axis = [0.1, 0.2, 0.4, 0.6, 0.8]

        if not evaluate_upstream:
            metrics = ["accuracy_groundtruth", "accuracy_imputer", "accuracy_meanimpute", "recall_groundtruth",
                       "recall_imputer", "recall_meanimpute", "f1_groundtruth", "f1_imputer", "f1_meanimpute"]

        model, loaded_cache, bar_plot, b_task = "none", False, False, "classification"

        if not generate_plot:
            plots = False
            inner_plots = False

        if not run_downstream:
            evaluate_upstream = True
            classifiers = ["no-model"]
            model = classifiers[0]

        directory_now = datetime.datetime.now()
        directory_time = directory_now.strftime("%y_%m_%d_%H_%M_%S")
        save_dir = save_dir + "/" + "benchmark_classifiers_" + report_title + "_" + directory_time

        if nbr_series is None:
            nbr_series = 10000000
        if nbr_vals is None:
            nbr_vals = 10000000

        benchmark_time = time.time()

        patterns = [
            "aligned_series" if x == "seqn"
            else "aligned_timestamps" if x == "blks"
            else x
            for x in patterns
        ]

        definition_of_exp = f"\nThe benchmark has been called (classification):\n\tclassifiers: {classifiers}\n\talgorithms: {algorithms}\n\tdatasets: {datasets}\n\tpatterns: {patterns}\n\tmissing_percentages: {x_axis}\n\tmetrics: {metrics}\n\tnormalizer: {normalizer}\n\tcontamination_by_class: {contamination_by_class}\n\timputation_by_class: {imputation_by_class}\n\tfixed_rate: {fixed_rate}\n\tto_cache: {to_cache}\n\tuse_cache: {use_cache}\n\tevaluate_upstream: {evaluate_upstream}\n\trun_downstream: {run_downstream}\n\treferential: {referential}\n\truns: {runs}\n\tnumber max series: {nbr_series}\n\tnumber max values: {nbr_vals}\n\n"
        print(definition_of_exp)

        if normalizer == "denorm":
            denormalization = True
            normalizer = "z_score"
        else:
            denormalization = False

        # =============================================================================================================
        # BENCHMARK
        # =============================================================================================================
        for i_run in range(0, abs(runs)):  # for each run of the benchmark

            for dataset in datasets:  # for each dataset (classification)
                runs_plots_scores = {}
                block_size_mcar = 10
                y_p_size = max(4, len(algorithms) * 0.275)

                # -----------------------------------------------------------------------------------------------------
                # DATASET LOADING
                # -----------------------------------------------------------------------------------------------------
                ts = TimeSeries(verbose=verbose)
                ts.load_classify_dataset(dataset, normalizer=normalizer, to_numpy=True, test_imputation=test_imputation,
                                         nbr_series=nbr_series, nbr_val=nbr_vals, verbose=verbose)
                classify_sets = ts.load_proper_output_classify()
                X_test, y_train, y_test, _, _, test_imputation, _, dataset = classify_sets
                N, M = ts.data.shape
                ts_m = None
                if M <= 0:
                    raise ValueError(f"The dataset loaded has no series (series {M}).")

                for pattern in patterns:  # for each pattern e.g, mcar

                    for algorithm in algorithms:  # for each imputation algorithms

                        print(
                            f"{algorithm} is tested with {pattern} on {dataset}, started at {time.strftime('%Y-%m-%d %H:%M:%S')}.")

                        for incx, x in enumerate(x_axis):  # for each missingness rate
                            ts_m = None

                            try:
                                start_time_imputation = time.time()

                                # ------------------------------------------------------------------------------------
                                # CACHING HANDLING
                                # ------------------------------------------------------------------------------------
                                if use_cache:
                                    name, dir_cache = utils.prepare_caching(dataset=dataset, algorithm=algorithm,
                                                                            pattern=pattern, x=x,
                                                                            contamination_by_class=contamination_by_class,
                                                                            imputation_by_class=imputation_by_class,
                                                                            fixed_rate=fixed_rate)

                                    cache_path_recov = os.path.join(dir_cache, name + "_recov.txt")
                                    cache_path_missing = os.path.join(dir_cache, name + "_miss.txt")
                                    ts_r = TimeSeries(verbose=False)
                                    ts_m = TimeSeries(verbose=False)

                                    if os.path.exists(cache_path_recov) and os.path.exists(cache_path_missing):
                                        ts_r.load_series(cache_path_recov, normalizer=None, verbose=False)
                                        ts_m.load_series(cache_path_missing, normalizer=None, verbose=False)
                                        imputer_tr = utils.config_impute_algorithm(incomp_data=ts_m.data,
                                                                                   algorithm=algorithm, task=b_task,
                                                                                   verbose=verbose)
                                        imputer_tr.recov_data = ts_r.data
                                        loaded_cache = True
                                        print(
                                            f"\tthe imputed matrix has been loaded from the cache {ts_r.data.shape}: {name}")
                                    else:
                                        loaded_cache = False

                                # ------------------------------------------------------------------------------------
                                # CONTAMINATION AND IMPUTATION
                                # ------------------------------------------------------------------------------------
                                if not loaded_cache and not bypass_error:
                                    cardinality_imputation_s = time.time()

                                    # --------------------------------------------------------------------------------
                                    # PIPELINE WITH CONTAMINATION BY DATASET
                                    # --------------------------------------------------------------------------------
                                    if not contamination_by_class:
                                        from imputegap.recovery.contamination import GenGap

                                        if verbose:
                                            print("\n\tcontamination by dataset")

                                        if pattern == "aligned_series" or pattern == "seqn":
                                            ts_m = GenGap.aligned(ts.data, rate_dataset=x, rate_series=fixed_rate,
                                                                  offset=0.05, verbose=verbose)
                                            if imputation_by_class:
                                                if verbose:
                                                    print("\n\t\timputation by class")
                                                imputer_tr, _, _ = utils.prepare_series_by_class(labels=y_train,
                                                                                                 raw_data=ts.data.copy(),
                                                                                                 imputer=algorithm,
                                                                                                 params=None, ts_m=ts_m,
                                                                                                 rate_dataset=x,
                                                                                                 rate_series=fixed_rate,
                                                                                                 task=b_task,
                                                                                                 verbose=verbose)
                                            else:
                                                if verbose:
                                                    print("\n\t\timputation by dataset")
                                                imputer_tr = utils.config_impute_algorithm(incomp_data=ts_m,
                                                                                           algorithm=algorithm,
                                                                                           task=b_task, verbose=verbose)
                                                imputer_tr.impute()

                                        elif pattern == "aligned_timestamps" or pattern == "blks":
                                            ts_m = GenGap.aligned(ts.data, rate_dataset=fixed_rate, rate_series=x,
                                                                  offset=0.05, verbose=verbose)

                                            if imputation_by_class:
                                                if verbose:
                                                    print("\n\t\timputation by class")
                                                imputer_tr, _, _ = utils.prepare_series_by_class(labels=y_train,
                                                                                                 raw_data=ts.data.copy(),
                                                                                                 imputer=algorithm,
                                                                                                 params=None, ts_m=ts_m,
                                                                                                 rate_dataset=fixed_rate,
                                                                                                 rate_series=x,
                                                                                                 task=b_task,
                                                                                                 verbose=verbose)
                                            else:
                                                if verbose:
                                                    print("\n\t\timputation by dataset")
                                                imputer_tr = utils.config_impute_algorithm(incomp_data=ts_m,
                                                                                           algorithm=algorithm,
                                                                                           task=b_task, verbose=verbose)
                                                imputer_tr.impute()


                                        else:
                                            ts_m = GenGap.mcar(ts.data, rate_dataset=fixed_rate, rate_series=x,
                                                               block_size=block_size_mcar, seed=True, offset=0.05,
                                                               verbose=False)

                                            if imputation_by_class:
                                                if verbose:
                                                    print("\n\t\timputation by class")
                                                imputer_tr, _, _ = utils.prepare_series_by_class(labels=y_train,
                                                                                                 raw_data=ts.data.copy(),
                                                                                                 imputer=algorithm,
                                                                                                 params=None, ts_m=ts_m,
                                                                                                 rate_dataset=fixed_rate,
                                                                                                 rate_series=x,
                                                                                                 task=b_task,
                                                                                                 verbose=verbose)
                                            else:
                                                if verbose:
                                                    print("\n\t\timputation by dataset")
                                                imputer_tr = utils.config_impute_algorithm(incomp_data=ts_m,
                                                                                           algorithm=algorithm,
                                                                                           task=b_task, verbose=verbose)
                                                imputer_tr.impute()

                                        miss = ts_m

                                    # -----------------------------------------------------------------------------
                                    # PIPELINE WITH CONTAMINATION BY CLASS
                                    # -----------------------------------------------------------------------------
                                    else:
                                        if verbose:
                                            print("\n\tcontamination by class")

                                        if pattern == "aligned_series":
                                            if verbose:
                                                print("\n\t\timputation by class")
                                            imputer_tr, _, miss = utils.prepare_series_by_class(labels=y_train,
                                                                                                raw_data=ts.data.copy(),
                                                                                                imputer=algorithm,
                                                                                                params=None,
                                                                                                pattern="aligned",
                                                                                                rate_dataset=x,
                                                                                                rate_series=fixed_rate,
                                                                                                offset=0.05,
                                                                                                task=b_task,
                                                                                                verbose=verbose,
                                                                                                imputation_by_class=imputation_by_class)
                                        elif pattern == "aligned_timestamps":
                                            if verbose:
                                                print("\n\t\timputation by class")
                                            imputer_tr, _, miss = utils.prepare_series_by_class(labels=y_train,
                                                                                                raw_data=ts.data.copy(),
                                                                                                imputer=algorithm,
                                                                                                params=None,
                                                                                                pattern="aligned",
                                                                                                rate_dataset=fixed_rate,
                                                                                                rate_series=x,
                                                                                                offset=0.05,
                                                                                                task=b_task,
                                                                                                imputation_by_class=imputation_by_class,
                                                                                                verbose=verbose)
                                        else:
                                            if verbose:
                                                print("\n\t\timputation by class")
                                            imputer_tr, _, miss = utils.prepare_series_by_class(labels=y_train,
                                                                                                raw_data=ts.data.copy(),
                                                                                                imputer=algorithm,
                                                                                                params=None,
                                                                                                pattern=pattern,
                                                                                                rate_dataset=fixed_rate,
                                                                                                rate_series=x,
                                                                                                offset=0.05,
                                                                                                task=b_task,
                                                                                                imputation_by_class=imputation_by_class,
                                                                                                verbose=verbose)

                                    # ---------------------------------------------------------------------------------
                                    # METRICS AND RESULTS
                                    # ---------------------------------------------------------------------------------
                                    cardinality_imputation_e = time.time()
                                    cardinality_imputation = (
                                                                         cardinality_imputation_e - cardinality_imputation_s) * 1000

                                    log_line = (f"NATE's LOG::::: {cardinality_imputation=} / "
                                                f"[{imputation_by_class=} - {contamination_by_class=}] - "
                                                f"[{x=}] - ({dataset}) for {algorithm}")

                                    with open("nate_log_upstream_rt.txt", "a", encoding="utf-8") as f:
                                        f.write(log_line + "\n")

                                if (not loaded_cache and not bypass_error) or (loaded_cache and bypass_error) or (
                                        loaded_cache and not bypass_error):
                                    end_time_imputation = time.time()

                                    dataset_s = dataset
                                    if "-" in dataset:
                                        dataset_s = dataset.replace("-", "")
                                    save_dir_plot = save_dir + "/" + dataset_s + "/" + pattern + "/recovery/"
                                    cont_rate = int(x * 100)

                                    # ---------------------------------------------------------------------------------
                                    # DOWNSTREAM - CLASSIFICATION
                                    # ---------------------------------------------------------------------------------
                                    if run_downstream:
                                        if verbose:
                                            print(f"\n\tmodel downstream launched: {run_downstream = }\t{loaded_cache=} - {bypass_error=}")

                                        imputer_tr.verbose = verbose
                                        imputer_tr.task = b_task

                                        for model in classifiers:

                                            artifact = Artifact(dataset=dataset_s,
                                                                model=model,
                                                                algorithm=algorithm,
                                                                pattern=pattern,
                                                                x=x,
                                                                contamination_by_class=contamination_by_class,
                                                                imputation_by_class=imputation_by_class,
                                                                fixed_x=fixed_rate)

                                            classifier = Classifier(task="classify",
                                                                    model=model,
                                                                    X_test=classify_sets[0],
                                                                    y_train=classify_sets[1],
                                                                    y_test=classify_sets[2],
                                                                    referential=referential,
                                                                    do_testset_imputation=False,
                                                                    algorithm=imputer_tr.algorithm,
                                                                    dataset=classify_sets[7],
                                                                    plots=False,
                                                                    use_cache=use_cache,
                                                                    to_cache=to_cache,
                                                                    artifact=artifact,
                                                                    bypass_error=bypass_error,
                                                                    v2=v2)

                                            downstream_analyser = imputer_tr
                                            ground_matrix = ts.data

                                            if denormalization:
                                                downstream_analyser = imputer_tr
                                                downstream_analyser.recov_data = ts.denormalize(
                                                    values=imputer_tr.recov_data)
                                                downstream_analyser.incomp_data = ts.denormalize(
                                                    values=imputer_tr.incomp_data)
                                                ground_matrix = ts.denormalize(values=ts.data)

                                            start_time_downstream = time.time()
                                            downstream_analyser.verbose = verbose
                                            downstream_analyser.score(ground_matrix, downstream_analyser.recov_data,
                                                                      downstream=classifier)
                                            imputer_tr.downstream_metrics = downstream_analyser.downstream_metrics
                                            end_time_downstream = time.time()

                                            time_downstream = (end_time_downstream - start_time_downstream) * 1000
                                            if time_downstream < 1:
                                                time_downstream = 1

                                            log_line_d = (f"NATE's LOG::::: {time_downstream=} / "
                                                          f"[{imputation_by_class=} - {contamination_by_class=}] - "
                                                          f"[{x=} - ({dataset}) for {algorithm} inside {model}")

                                            with open("nate_log_downstream_rt.txt", "a", encoding="utf-8") as f:
                                                f.write(log_line_d + "\n")

                                            if not evaluate_upstream:
                                                imputer_tr.score(input_data=ts.data, recov_data=imputer_tr.recov_data,
                                                                 verbose=False)
                                                imputer_tr.downstream_metrics["RMSE"] = imputer_tr.metrics["RMSE"]
                                                imputer_tr.downstream_metrics["RUNTIME"] = time_downstream
                                                runs_plots_scores.setdefault(str(dataset_s), {}).setdefault(
                                                    str(pattern), {}).setdefault(str(algorithm), {}).setdefault(
                                                    str(model), {})[str(x)] = {"scores": imputer_tr.downstream_metrics}

                                            # cleanup
                                            del classifier
                                            gc.collect()
                                            if torch.cuda.is_available():
                                                torch.cuda.empty_cache()
                                                torch.cuda.ipc_collect()

                                    if evaluate_upstream:
                                        imputer_tr.score(input_data=ts.data, recov_data=imputer_tr.recov_data,
                                                         verbose=False)

                                    # ---------------------------------------------------------------------------------
                                    # CACHING RESULTS
                                    # ---------------------------------------------------------------------------------
                                    if to_cache and not loaded_cache:
                                        if not np.isnan(imputer_tr.recov_data).any():
                                            name, dir_cache = utils.prepare_caching(dataset=dataset,
                                                                                    algorithm=algorithm,
                                                                                    pattern=pattern, x=x,
                                                                                    contamination_by_class=contamination_by_class,
                                                                                    imputation_by_class=imputation_by_class,
                                                                                    fixed_rate=fixed_rate)
                                            utils.ts_caching_save(data=imputer_tr.recov_data, type="recov",
                                                                  artifact=classify_sets[1], name=name,
                                                                  dir_cache=dir_cache)
                                            utils.ts_caching_save(data=miss, type="miss", artifact=None, name=name,
                                                                  dir_cache=dir_cache)
                                            print(f"\tcaching stored for recovery and missing matrix: {name}")
                                        else:
                                            print(
                                                f"\t\t\tcaching FAILED, the recovery matrix has NaNs on it...: {name}")

                                    # ---------------------------------------------------------------------------------
                                    # SAVE RESULTS
                                    # ---------------------------------------------------------------------------------
                                    if "*" not in metrics and "all" not in metrics:
                                        imputer_tr.metrics = {k: imputer_tr.metrics[k] for k in metrics if
                                                              k in imputer_tr.metrics}

                                    time_imputation = (end_time_imputation - start_time_imputation) * 1000
                                    time_imputation = max(time_imputation, 1)
                                    log_time_imputation = math.log10(time_imputation) if time_imputation > 0 else None

                                    if run_downstream:
                                        f1_vals = [
                                            v for k, v in imputer_tr.downstream_metrics.items()
                                            if "f1" in k.lower()
                                               and "groundtruth" not in k.lower()
                                               and "mean-impute" not in k.lower()
                                        ]

                                        f1_value = f1_vals[0] if f1_vals else None
                                        imputer_tr.metrics["RUNTIME"] = time_imputation
                                        imputer_tr.metrics["RUNTIME_LOG"] = log_time_imputation
                                        imputer_tr.metrics["DOWNSTREAM_ACC"] = f1_value
                                    else:
                                        imputer_tr.metrics["RUNTIME"] = time_imputation
                                        imputer_tr.metrics["RUNTIME_LOG"] = log_time_imputation
                                        imputer_tr.metrics["DOWNSTREAM_ACC"] = np.nan

                                    if inner_plots:
                                        try:
                                            ts.plot(input_data=ts.data, incomp_data=imputer_tr.incomp_data,
                                                    recov_data=imputer_tr.recov_data, nbr_series=6, subplot=True,
                                                    algorithm=imputer_tr.algorithm, cont_rate=str(cont_rate),
                                                    display=False, save_path=save_dir_plot, verbose=False)
                                        except Exception as e:
                                            print(f"Error during plotting: {e}")
                                    if evaluate_upstream:
                                        runs_plots_scores.setdefault(str(dataset_s), {}).setdefault(str(pattern),
                                                                                                    {}).setdefault(
                                            str(algorithm), {}).setdefault(str(model), {})[str(x)] = {
                                            "scores": imputer_tr.metrics}

                                else:
                                    print(f"\t\t{bypass_error=} / {loaded_cache=}")
                                    imputer_tr, runs_plots_scores = self._benchmark_error("bypass", None,
                                                                                          runs_plots_scores, ts_m,
                                                                                          classifiers, dataset,
                                                                                          algorithm, pattern, x,
                                                                                          save_dir, verbose)

                            except Exception as e:
                                imputer_tr, runs_plots_scores = self._benchmark_error("error", e, runs_plots_scores,
                                                                                      ts_m, classifiers, dataset,
                                                                                      algorithm, pattern, x, save_dir,
                                                                                      verbose)

                        print("done!\n\n")

                run_storage.append(runs_plots_scores)

        plt.close('all')  # Close all open figures

        # ---------------------------------------------------------------------------------
        # SAVE PLOTS, REPORTS, LOGS, AND ...
        # ---------------------------------------------------------------------------------
        for x, m in enumerate(reversed(metrics)):
            scores_list, algos, sets = self.avg_results_task(*run_storage, metric=m, upstream=evaluate_upstream)
            try:
                if generate_plot:
                    _ = self.generate_heatmap(scores_list=scores_list, algos=algos, sets=sets, metric=m,
                                              save_dir=save_dir, display=False)
                if not plots:
                    plt.close('all')
            except Exception as e:
                print(f"Error during plotting: {e}")

        benchmark_end = time.time()
        total_time_benchmark = round(benchmark_end - benchmark_time, 4)
        h = int(total_time_benchmark // 3600)
        m = int((total_time_benchmark % 3600) // 60)
        s = total_time_benchmark % 60
        print(f"\n> logs: benchmark - Execution Time: {total_time_benchmark} seconds,  ({h:02d}:{m:02d}:{s:06.3f} hh:mm:ss)\n")

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "runtime.log")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(save_path, "a") as file:
            file.write(f"{timestamp} | logs: benchmark - Execution Time: {total_time_benchmark} seconds,  ({h:02d}:{m:02d}:{s:06.3f} hh:mm:ss)\n")
        verb = False

        for scores in run_storage:
            all_keys = list(scores.keys())
            dataset_name = str(all_keys[0])
            save_dir_agg_set = save_dir + "/" + dataset_name

            try:
                self.generate_reports_txt(runs_plots_scores=scores, save_dir=save_dir_agg_set, given_dataset=dataset_name, metrics=metrics, rt=total_time_benchmark, upstream=evaluate_upstream, run=-1)
            except Exception as e:
                print(f"Error during file report generation: {e}")
            try:
                if generate_plot:
                    self.generate_plots(runs_plots_scores=scores, ticks=x_axis, metrics=metrics, subplot=True,
                                        y_size=y_p_size, save_dir=save_dir_agg_set, upstream=evaluate_upstream,
                                        display=verb)
                if not plots:
                    plt.close('all')
            except Exception as e:
                print(f"Error during plotting: {e}")

        try:
            self.generate_reports_summary(run_of_values=run_storage, save_dir=save_dir, metrics=metrics,
                                          rt=total_time_benchmark, run=-1, title=report_title,
                                          upstream=evaluate_upstream)
        except Exception as e:
            print(f"Error during file report generation: {e}")

        try:
            if generate_plot and bar_plot:
                self.generate_avg_bar_plot(results=run_storage, save_path=save_dir, type="classifier-down",
                                           verbose=verbose, plots=plots, upstream=evaluate_upstream)
            if not plots:
                plt.close('all')
        except Exception as e:
            print(f"Error during plotting: {e}")

        try:
            self.save_into_excel(run_of_values=run_storage,
                                 xlsx_path=os.path.join(save_dir, f"_summary_{report_title}.xlsx"),
                                 upstream=evaluate_upstream, verbose=verbose)
        except Exception as e:
            print(f"Error during report generation: {e}")

        print(f"\nThe results are saved in : {save_dir}\n")
        self.list_results = run_storage

        if evaluate_upstream:
            self.aggregate_results = scores_list

        save_def = os.path.join(save_dir, "experimentation_setup.log")

        try:
            with open(save_def, "w") as file:
                file.write(definition_of_exp)
        except Exception as e:
            print(f"Error during report generation: {e}")

    def eval_downstream_forecasting(self, forecasters=None, horizon=12, algorithms=None, datasets=None, patterns=None, x_axis=None, metrics=None, save_dir="./imputegap_assets/benchmark", runs=1, normalizer="z_score", report_title="", nbr_series=10000, nbr_vals=1000, to_cache=True, use_cache=True, run_downstream=False, fixed_rate=0.2, evaluate_upstream=True, bypass_error=False, referential=True, generate_plot=True, plots=True, inner_plots=True, verbose=False):
        """
        Benchmark imputation/recovery algorithms using a downstream forecasting task across datasets, missingness patterns, and missingness rates.

        This routine runs an end-to-end benchmark loop:

        1) Load each forecasting dataset with the horizon

        2) Generate missingness either globally or class-conditionally (controlled by `contamination_class`) for multiple missingness rates (`x_axis`) and patterns (`patterns`).

        3) Impute/recover the contaminated training data (and optionally test data) using each specified algorithm.

        4) Score the recovered matrix with reconstruction metrics (e.g., RMSE/MAE/etc.) and a downstream forecasting metric extracted from `imputer_tr.downstream_metrics`

        5) Save plots, logs, and aggregated reports (heatmaps, line plots, summary tables).

        The benchmark results are stored per run in a nested dictionary structure: results[dataset][pattern][algorithm][forecaster][missing_rate] = {"scores": imputer_tr.metrics}


        Parameters
        ----------
        forecasters : list[str], optional
            Forecasting model identifiers to evaluate downstream. For all forecasters please use `utils.list_of_forecasters()`. If None, default is ["chronos"].

        horizon: int, optional
            Value of the horizon for the prediction, e.g, 12

        algorithms : list[str], optional
            Imputation/recovery algorithms to benchmark. If "*" / "all" is used, it is replaced by `utils.list_of_algorithms()`. If None, default is ["cdrec"].

        datasets : list[str], optional
            Forecasting datasets to benchmark. If None, default is ["paris"]. For all use: utils.get_dataset_forecasters(directory='datasets/forecast/')

        patterns : list[str], optional
            Missingness/contamination patterns (e.g., "mcar", "aligned_series", "aligned_timestamps"). If None, default is ["mcar"].

        x_axis : list[float], optional
            Missingness rates used for contamination (interpreted as proportions; e.g. 0.1 = 10%). If None, default is [0.1, 0.2, 0.4, 0.6, 0.8].

        metrics : list[str], optional
            Subset of reconstruction metrics to keep in `imputer_tr.metrics`. If "*" / "all" is present, uses `utils.list_of_metrics()`. If None, default is ["*"].

        save_dir : str, optional
            Base output directory for saving benchmark artifacts. A timestamped subdirectory is created. Default is "./imputegap_assets/benchmark".

        runs : int, optional
            Number of repeated benchmark runs. The absolute value is used in the loop. Default is 1.

        normalizer : str, optional
            Normalization method used when loading datasets (passed to `ts.load_classify_dataset`). Default is "z_score".

        contamination_by_class : bool, optional
            If True, contamination/missingness is generated in a class-conditional manner (each class can be contaminated independently > advised).
            If False, a single global missingness mask is generated first and then applied across classes.
            Default is True.

        imputation_by_class : bool, optional
            If True, runs the imputation/recovery step separately per class partition (class-wise imputation> advised).
            If False, imputation is performed on the full contaminated training matrix.
            Default is True.

        report_title : str, optional
            Optional title string inserted into the output directory name and summary report. Default is "".

        nbr_series : int or None, optional
            Maximum number of series to include in evaluation (used by dataset loader). If None, treated as very
            large. Default is 10000.

        nbr_vals : int or None, optional
            Maximum number of values/timestamps to include in evaluation (used by dataset loader). If None,
            treated as very large. Default is 1000.

        to_cache : bool, optional
            If True, stores intermediate benchmark artifacts/results into cache (when supported by your pipeline),
            so repeated runs can be faster/reproducible. Default is True.

        use_cache : bool, optional
            If True, attempts to load cached intermediate artifacts/results instead of recomputing them.
            Default is True.

        run_downstream : bool, optional
            If True, executes downstream classification evaluation and stores downstream metrics (e.g. accuracy)
            alongside reconstruction metrics. If False, skips downstream evaluation.
            Default is False.

        fixed_rate : float, optional
            Fixed missingness rate used in contexts where a single rate is required (e.g., some per-class routines,
            baseline comparisons, or when overriding dynamic rate selection). Default is 0.2.

        evaluate_upstream : bool, optional
            If True, computes and records upstream/reconstruction metrics (e.g., RMSE/MAE/MI/CORRELATION).
            If False, computes and records downstream metrics.

        bypass_error : bool, optional
            If True, this variable will put automatically NaN values if an error is encountered.
            Put to true if you want to get the results of all success imputer

        generate_plot: bool, optional
            If True, generates and saves plots (e.g., heatmaps/line plots) for benchmark results.
            Default is True.

        plots : bool, optional
            If True, display plots (e.g., heatmaps/line plots) for benchmark results.
            Default is True.

        inner_plots : bool, optional
            If True, generates and saves every steps plots for each individual datasets for benchmark results.
            Default is True.

        referential: bool, optional
            If True, run the referential algorithm to compare the benchmark results (downstream)

        verbose : bool, optional
            Verbosity flag controlling detailed logging and plotting behavior. Default is False.

        Example
        -------
        .. code-block:: python

            from imputegap.recovery.benchmark import Benchmark
            from imputegap.tools import utils

            bench = Benchmark()
            bench.eval_downstream_forecasting(forecasters=["exp-smoothing"],
                                              horizon=[12],
                                              algorithms=utils.list_of_top_cleanimp_for_algorithms(),
                                              datasets=utils.get_dataset_forecasters(directory='datasets/forecast/'),
                                              patterns=["mcar", "aligned_series", "aligned_timestamps"],
                                              x_axis=[0.1, 0.2, 0.4, 0.6, 0.8],
                                              metrics=["RMSE", "DOWNSTREAM_SMAPE", "MI", "CORRELATION", "RUNTIME"],
                                              normalizer="z-score",
                                              report_title="cleanimp_benchmark_for_up",
                                              bypass_error=False,
                                              to_cache=True,
                                              use_cache=True,
                                              run_downstream=False,
                                              evaluate_upstream=True,
                                              fixed_rate=0.2,
                                              verbose=False,
                                              inner_plots=False,
                                              plots=False,
                                              generate_plot=True,
                                              referential=False,
                                              nbr_series=10000,
                                              nbr_vals=10000)
        """

        # =============================================================================================================
        # DATA AND CONFIGURATION MANAGEMENT
        # =============================================================================================================
        from imputegap.recovery.downstream import Forecaster
        from imputegap.recovery.downstream import Artifact
        from imputegap.recovery.contamination import GenGap
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import gc
        try:
            import torch
        except ImportError:
            torch = None
        run_storage = []

        if forecasters is None:
            forecasters = ["chronos"]
        if algorithms is None:
            algorithms = ["cdrec"]
        if datasets is None:
            datasets = ["paris"]
        if patterns is None:
            patterns = ["mcar"]
        if x_axis is None:
            x_axis = [0.1, 0.2, 0.4, 0.6, 0.8]
        if metrics is None:
            metrics = ["*"]

        if not isinstance(algorithms, list):
            raise TypeError(f"'algorithms' must be a list, but got {type(algorithms).__name__}")
        if not isinstance(datasets, list):
            raise TypeError(f"'datasets' must be a list, but got {type(datasets).__name__}")
        if not isinstance(patterns, list):
            raise TypeError(f"'patterns' must be a list, but got {type(patterns).__name__}")
        if not isinstance(x_axis, list):
            raise TypeError(f"'x_axis' must be a list, but got {type(x_axis).__name__}")
        if not isinstance(forecasters, list):
            raise TypeError(f"'forecasters' must be a list, but got {type(forecasters).__name__}")

        if metrics is None or ("*" in metrics or "all" in metrics):
            metrics = utils.list_of_metrics()
        if "*" in algorithms or "all" in algorithms:
            algorithms = utils.list_of_algorithms()
        if "*" in forecasters or "all" in forecasters:
            forecasters = utils.list_of_forecasters()
        if "*" in datasets or "all" in datasets:
            datasets = utils.get_dataset_forecasters(directory='datasets/forecast/')
        if "*" in patterns or "all" in patterns:
            patterns = ["mcar", "aligned_series", "aligned_timestamps"]
        if "*" in x_axis or "all" in x_axis:
            x_axis = [0.1, 0.2, 0.4, 0.6, 0.8]

        if not evaluate_upstream:
            metrics = ["mse_groundtruth", "mse_imputer", "mse_baseline",
                       "mae_groundtruth", "mae_imputer", "mae_baseline",
                       "smape_groundtruth", "smape_imputer", "smape_baseline"]

        model, loaded_cache = "none", False

        if not run_downstream:
            evaluate_upstream = True
            forecasters = ["no-model"]
            model = forecasters[0]

        directory_now = datetime.datetime.now()
        directory_time = directory_now.strftime("%y_%m_%d_%H_%M_%S")
        save_dir = save_dir + "/" + "benchmark_forecasters_" + report_title + "_" + directory_time
        horizons = [horizon]
        b_task = "forecasting"

        if nbr_series is None:
            nbr_series = 10000000
        if nbr_vals is None:
            nbr_vals = 10000000

        if not generate_plot:
            plots = False
            inner_plots = False

        benchmark_time = time.time()

        definition_of_exp = f"\nThe benchmark has been called (forecasting):\n\tforecasters: {forecasters}\n\thorizons: {horizons}\n\talgorithms: {algorithms}\n\tdatasets: {datasets}\n\tpatterns: {patterns}\n\tmissing_percentages: {x_axis}\n\tmetrics: {metrics}\n\tnormalizer: {normalizer}\n\tfixed_rate: {fixed_rate}\n\tto_cache: {to_cache}\n\tuse_cache: {use_cache}\n\tevaluate_upstream: {evaluate_upstream}\n\trun_downstream: {run_downstream}\n\tbypass_error: {bypass_error}\n\treferential: {referential}\n\truns: {runs}\n\tnumber max series: {nbr_series}\n\tnumber max values: {nbr_vals}\n\n"
        print(definition_of_exp)

        if normalizer == "denorm":
            denormalization = True
            normalizer = "z_score"
        else:
            denormalization = False

        # =============================================================================================================
        # BENCHMARK
        # =============================================================================================================
        for i_run in range(0, abs(runs)):
            for dataset in datasets:
                for horizon_ in horizons:
                    runs_plots_scores = {}
                    block_size_mcar = 10
                    y_p_size = max(4, len(algorithms) * 0.275)
                    s_horizon = str(horizon)

                    # -----------------------------------------------------------------------------------------------------
                    # DATASET LOADING
                    # -----------------------------------------------------------------------------------------------------
                    ts = TimeSeries(verbose=verbose)
                    ts.load_forecasting_dataset(dataset, normalizer=normalizer, verbose=verbose, horizon=horizon_, nbr_series=nbr_series, nbr_val=nbr_vals)
                    y_train, season = ts.load_proper_output_forecasting()
                    N, M = ts.data.shape
                    print(f"{dataset}:{ts.data.shape = }:{season=}-{horizon_=}")

                    if M <= 0:
                        raise ValueError(f"The dataset loaded has no series (series {M}).")

                    for pattern in patterns:  # for each pattern e.g, mcar

                        for algorithm in algorithms:  # for each algorithm e.g, cdrec

                            print(
                                f"{algorithm} is tested with {pattern} on {dataset}, started at {time.strftime('%Y-%m-%d %H:%M:%S')}.")

                            for incx, x in enumerate(x_axis):  # for each missingness rate e.g, 0.2

                                try:
                                    start_time_imputation = time.time()

                                    # ------------------------------------------------------------------------------------
                                    # CACHING HANDLING
                                    # ------------------------------------------------------------------------------------
                                    if use_cache:
                                        name, dir_cache = utils.prepare_caching(dataset=dataset, algorithm=algorithm, pattern=pattern, x=x, contamination_by_class=None, imputation_by_class=s_horizon, fixed_rate=fixed_rate)

                                        name = "_f_" + name
                                        cache_path_recov = os.path.join(dir_cache, name + "_recov.txt")
                                        cache_path_missing = os.path.join(dir_cache, name + "_miss.txt")
                                        ts_r = TimeSeries(verbose=False)
                                        ts_m = TimeSeries(verbose=False)

                                        if os.path.exists(cache_path_recov) and os.path.exists(cache_path_missing):
                                            ts_r.load_series(cache_path_recov, normalizer=None, verbose=False)
                                            ts_m.load_series(cache_path_missing, normalizer=None, verbose=False)
                                            imputer = utils.config_impute_algorithm(incomp_data=ts_m.data,
                                                                                    algorithm=algorithm,
                                                                                    verbose=verbose)
                                            imputer.recov_data = ts_r.data
                                            imputer.task = b_task
                                            loaded_cache = True
                                            print(
                                                f"\tthe imputed matrix (forecasting) has been loaded from the cache {ts_r.data.shape}: {name}")
                                        else:
                                            loaded_cache = False

                                    # ------------------------------------------------------------------------------------
                                    # CONTAMINATION OF THE DATA
                                    # ------------------------------------------------------------------------------------
                                    if not loaded_cache and not bypass_error:
                                        if pattern == "aligned_series" or pattern == "seqn":
                                            incomp_data = utils.config_contamination(ts=ts, pattern="aligned", dataset_rate=fixed_rate, series_rate=x, verbose=verbose, offset=0)
                                        elif pattern == "aligned_timestamps" or pattern == "blks":
                                            incomp_data = utils.config_contamination(ts=ts, pattern="aligned", dataset_rate=x, series_rate=fixed_rate, verbose=verbose, offset=0)
                                        else:
                                            incomp_data = utils.config_contamination(ts=ts, pattern=pattern, dataset_rate=fixed_rate, series_rate=x, block_size=block_size_mcar, verbose=verbose, offset=0)

                                        # MIRROR OF THE CONTAMINATION (FOR FORECASTING PURPOSE - START AT THE HORIZON)
                                        # ---------------------------------------------------------------------------------------
                                        miss = GenGap.mirror_gap(raw_data=ts.data, ts_m=incomp_data)

                                        try:
                                            # ------------------------------------------------------------------------------------
                                            # IMPUTATION OF THE DATA
                                            # ------------------------------------------------------------------------------------
                                            imputer = utils.config_impute_algorithm(incomp_data=miss, algorithm=algorithm, task=b_task, verbose=verbose)
                                            imputer.verbose = verbose
                                            imputer.task = b_task
                                            imputer.impute()

                                        except Exception as e:
                                            print(f"Error during imputation: {e}")

                                    if (not loaded_cache and not bypass_error) or (loaded_cache and bypass_error) or (
                                            loaded_cache and not bypass_error):
                                        end_time_imputation = time.time()

                                        dataset_s = dataset
                                        if "-" in dataset:
                                            dataset_s = dataset.replace("-", "")
                                        save_dir_plot = save_dir + "/" + dataset_s + "/" + pattern + "/recovery/"
                                        cont_rate = int(x * 100)

                                        # -----------------------------------------------------------------------------
                                        # DOWNSTREAM PIPELINE
                                        # -----------------------------------------------------------------------------
                                        if run_downstream:
                                            if verbose:
                                                print(f"\n\tmodel downstream launched: {run_downstream = }")

                                            imputer.verbose = verbose
                                            imputer.task = b_task

                                            for model in forecasters:

                                                if model.lower() == "arima" and dataset_s in ["czelan", "airq"]:
                                                    s_exception = True
                                                else:
                                                    s_exception = False

                                                if model == "chronos":
                                                    import logging
                                                    for name in ["pytorch_lightning",
                                                                 "pytorch_lightning.utilities.rank_zero", "lightning",
                                                                 "lightning.pytorch",
                                                                 "lightning.pytorch.utilities.rank_zero", ]:
                                                        logging.getLogger(name).setLevel(logging.ERROR)

                                                artifact = Artifact(dataset=dataset_s,
                                                                    model=model,
                                                                    algorithm=algorithm,
                                                                    pattern=pattern,
                                                                    x=x,
                                                                    contamination_by_class=None,
                                                                    imputation_by_class=s_horizon,
                                                                    fixed_x=fixed_rate,
                                                                    horizon=horizon_)

                                                forcaster = Forecaster(task="forecast",
                                                                       model=model,
                                                                       y_train=y_train,
                                                                       season=season,
                                                                       plots=False,
                                                                       use_cache=use_cache,
                                                                       to_cache=to_cache,
                                                                       artifact=artifact,
                                                                       referential=referential,
                                                                       bypass_error=bypass_error,
                                                                       s_exception=s_exception
                                                                       )

                                                downstream_analyser = imputer
                                                ground_matrix = ts.data

                                                if denormalization:
                                                    downstream_analyser.recov_data = ts.denormalize(
                                                        values=imputer.recov_data)
                                                    downstream_analyser.incomp_data = ts.denormalize(
                                                        values=imputer.incomp_data)
                                                    ground_matrix = ts.denormalize(values=ts.data)

                                                start_time_downstream = time.time()

                                                downstream_analyser.verbose = verbose
                                                downstream_analyser.score(ground_matrix, downstream_analyser.recov_data, downstream=forcaster)
                                                imputer.downstream_metrics = downstream_analyser.downstream_metrics
                                                end_time_downstream = time.time()

                                                time_downstream = (end_time_downstream - start_time_downstream) * 1000
                                                if time_downstream < 1:
                                                    time_downstream = 1

                                                if not evaluate_upstream:
                                                    imputer.score(input_data=ts.data, recov_data=imputer.recov_data,
                                                                  verbose=False)
                                                    imputer.downstream_metrics["RMSE"] = imputer.metrics["RMSE"]
                                                    imputer.downstream_metrics["RUNTIME"] = time_downstream
                                                    runs_plots_scores.setdefault(str(dataset_s), {}).setdefault(
                                                        str(pattern), {}).setdefault(str(algorithm), {}).setdefault(
                                                        str(model), {})[str(x)] = {"scores": imputer.downstream_metrics}

                                                # cleanup
                                                del forcaster
                                                gc.collect()
                                                if torch is not None and torch.cuda.is_available():
                                                    torch.cuda.empty_cache()
                                                    torch.cuda.ipc_collect()

                                        imputer.score(input_data=ts.data, recov_data=imputer.recov_data, verbose=False)

                                        # STORAGE OF THE CACHED RESULTS
                                        # -----------------------------------------------------------------------------
                                        if to_cache and not loaded_cache:
                                            if not np.isnan(
                                                    imputer.recov_data).any() and imputer.recov_data.shape == miss.shape:
                                                name, dir_cache = utils.prepare_caching(dataset=dataset,
                                                                                        algorithm=algorithm,
                                                                                        pattern=pattern, x=x,
                                                                                        contamination_by_class=None,
                                                                                        imputation_by_class=s_horizon,
                                                                                        fixed_rate=fixed_rate)
                                                name = "_f_" + name
                                                utils.ts_caching_save(data=imputer.recov_data, type="recov",
                                                                      artifact=None, name=name, dir_cache=dir_cache)
                                                utils.ts_caching_save(data=miss, type="miss", artifact=None, name=name,
                                                                      dir_cache=dir_cache)
                                                print(f"\tcaching stored for recovery and missing matrix: {name}")
                                            else:
                                                print(
                                                    f"\t\t\tcaching FAILED, the recovery matrix has NaNs on it...: {algorithm}")

                                        # RESULT AND METRICS
                                        # -----------------------------------------------------------------------------
                                        if "*" not in metrics and "all" not in metrics:
                                            imputer.metrics = {k: imputer.metrics[k] for k in metrics if
                                                               k in imputer.metrics}

                                        time_imputation = (end_time_imputation - start_time_imputation) * 1000
                                        time_imputation = max(time_imputation, 1)
                                        log_time_imputation = math.log10(time_imputation) if time_imputation > 0 else None

                                        if run_downstream:
                                            smape_vals = [
                                                v for k, v in imputer.downstream_metrics.items()
                                                if "smape" in k.lower()
                                                   and "groundtruth" not in k.lower()
                                                   and "mean-impute" not in k.lower()
                                            ]

                                            smape_value = smape_vals[0] if smape_vals else np.nan

                                            imputer.metrics["RUNTIME"] = time_imputation
                                            imputer.metrics["RUNTIME_LOG"] = log_time_imputation
                                            imputer.metrics["DOWNSTREAM_SMAPE"] = smape_value

                                        else:
                                            imputer.metrics["RUNTIME"] = time_imputation
                                            imputer.metrics["RUNTIME_LOG"] = log_time_imputation
                                            imputer.metrics["DOWNSTREAM_SMAPE"] = np.nan

                                        if inner_plots:
                                            try:
                                                ts.plot(input_data=ts.data, incomp_data=imputer.incomp_data,
                                                        recov_data=imputer.recov_data, nbr_series=6, subplot=True,
                                                        algorithm=imputer.algorithm, cont_rate=str(cont_rate),
                                                        display=False, save_path=save_dir_plot, verbose=False)
                                            except Exception as e:
                                                print(f"Error during plotting: {e}")

                                        if evaluate_upstream:
                                            runs_plots_scores.setdefault(str(dataset_s), {}).setdefault(str(pattern),
                                                                                                        {}).setdefault(
                                                str(algorithm), {}).setdefault(str(model), {})[str(x)] = {
                                                "scores": imputer.metrics}

                                    else:
                                        print(f"\t\t{bypass_error=} / {loaded_cache=}")
                                        imputer_tr, runs_plots_scores = self._benchmark_error("bypass", None,
                                                                                              runs_plots_scores, ts_m,
                                                                                              forecasters, dataset,
                                                                                              algorithm, pattern, x,
                                                                                              save_dir, verbose)

                                    del imputer
                                    gc.collect()
                                    if torch is not None and torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                        torch.cuda.ipc_collect()

                                except Exception as e:
                                    import traceback

                                    print(f"\nERROR: {type(e).__name__}: {e}")
                                    traceback.print_exc()


                                    ts_m = np.zeros(len(ts.data))


                                    imputer_tr, runs_plots_scores = self._benchmark_error("error", e, runs_plots_scores,
                                                                                          ts_m, forecasters, dataset,
                                                                                          algorithm, pattern, x,
                                                                                          save_dir, verbose)

                            print("done!\n\n")

                    run_storage.append(runs_plots_scores)

        plt.close('all')  # Close all open figures

        # PLOTTING AND RESULTS
        # -----------------------------------------------------------------------------
        for x, m in enumerate(reversed(metrics)):
            scores_list, algos, sets = self.avg_results_task(*run_storage, metric=m, upstream=evaluate_upstream)
            try:
                if generate_plot:
                    _ = self.generate_heatmap(scores_list=scores_list, algos=algos, sets=sets, metric=m, save_dir=save_dir, display=False)
                if not plots:
                    plt.close('all')
            except Exception as e:
                print(f"Error during plotting: {e}")

        benchmark_end = time.time()
        total_time_benchmark = round(benchmark_end - benchmark_time, 4)
        h = int(total_time_benchmark // 3600)
        m = int((total_time_benchmark % 3600) // 60)
        s = total_time_benchmark % 60
        print(
            f"\n> logs: benchmark - Execution Time: {total_time_benchmark} seconds,  ({h:02d}:{m:02d}:{s:06.3f} hh:mm:ss)\n")

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "runtime.log")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(save_path, "a") as file:
            file.write(
                f"{timestamp} | logs: benchmark - Execution Time: {total_time_benchmark} seconds,  ({h:02d}:{m:02d}:{s:06.3f} hh:mm:ss)\n")
        verb = False

        for scores in run_storage:
            all_keys = list(scores.keys())
            dataset_name = str(all_keys[0])
            save_dir_agg_set = save_dir + "/" + dataset_name

            try:
                if generate_plot:
                    self.generate_reports_txt(runs_plots_scores=scores, save_dir=save_dir_agg_set, given_dataset=dataset_name,
                                              metrics=metrics, rt=total_time_benchmark, upstream=evaluate_upstream,
                                              run=-1)
            except Exception as e:
                print(f"Error during file report generation: {e}")
            try:
                if generate_plot:
                    self.generate_plots(runs_plots_scores=scores, ticks=x_axis, metrics=metrics, subplot=True,
                                        y_size=y_p_size, save_dir=save_dir_agg_set, upstream=evaluate_upstream,
                                        display=verb)
                if not plots:
                    plt.close('all')
            except Exception as e:
                print(f"Error during plotting: {e}")

        try:
            self.generate_reports_summary(run_of_values=run_storage, save_dir=save_dir, metrics=metrics,
                                          rt=total_time_benchmark, run=-1, title=report_title,
                                          upstream=evaluate_upstream)
        except Exception as e:
            print(f"Error during file report generation: {e}")

        try:
            self.save_into_excel(run_of_values=run_storage,
                                 xlsx_path=os.path.join(save_dir, f"_summary_{report_title}.xlsx"),
                                 upstream=evaluate_upstream, verbose=verbose)
        except Exception as e:
            print(f"Error during report generation: {e}")

        print(f"\nThe results are saved in : {save_dir}\n")
        self.list_results = run_storage

        if evaluate_upstream:
            self.aggregate_results = scores_list

        save_def = os.path.join(save_dir, "experimentation_setup.log")

        try:
            with open(save_def, "w") as file:
                file.write(definition_of_exp)
        except Exception as e:
            print(f"Error during report generation: {e}")


    def eval_imputation(self, algorithms=None, datasets=None, patterns=None, x_axis=None, metrics=None, save_dir="./imputegap_assets/benchmark", normalizer="z_score", report_title="", nbr_series=1000, nbr_vals=1000, to_cache=True, use_cache=True, fixed_rate=0.2, bypass_error=False, generate_plot=True, plots=True, inner_plots=True, verbose=False):
        """
        Benchmark imputation/recovery algorithms across datasets, missingness patterns, and missingness rates.

        This routine runs an end-to-end benchmark loop:

        1) Load each dataset and pre-process them

        2) Generate missingness for multiple missingness rates (`x_axis`) and patterns (`patterns`).

        3) Impute/recover the contaminated training data (and optionally test data) using each specified algorithm.

        4) Score the recovered matrix with reconstruction metrics (e.g., RMSE/MAE/etc.)

        5) Save plots, logs, and aggregated reports (heatmaps, line plots, summary tables).

        The benchmark results are stored per run in a nested dictionary structure: results[dataset][pattern][algorithm][missing_rate] = {"scores": imputer_tr.metrics}


        Parameters
        ----------
        algorithms : list[str], optional
            Imputation/recovery algorithms to benchmark. If "*" / "all" is used, it is replaced by `utils.list_of_algorithms()`. If None, default is ["cdrec"].

        datasets : list[str], optional
            Forecasting datasets to benchmark. Default is ["paris"]. For all use: utils.get_dataset_forecasters(directory='datasets/forecast/')

        patterns : list[str], optional
            Missingness/contamination patterns (e.g., "mcar", "aligned_series", "aligned_timestamps"). If None, default is ["mcar"].

        x_axis : list[float], optional
            Missingness rates used for contamination (interpreted as proportions; e.g. 0.1 = 10%). If None, default is [0.1, 0.2, 0.4, 0.6, 0.8].

        metrics : list[str], optional
            Subset of reconstruction metrics to keep in `imputer_tr.metrics`. If "*" / "all" is present, uses `utils.list_of_metrics()`. If None, default is ["*"].

        save_dir : str, optional
            Base output directory for saving benchmark artifacts. A timestamped subdirectory is created. Default is "./imputegap_assets/benchmark".

        normalizer : str, optional
            Normalization method used when loading datasets (passed to `ts.load_classify_dataset`). Default is "z_score".

        report_title : str, optional
            Optional title string inserted into the output directory name and summary report. Default is "".

        nbr_series : int or None, optional
            Maximum number of series to include in evaluation (used by dataset loader). If None, treated as very
            large. Default is 10000.

        nbr_vals : int or None, optional
            Maximum number of values/timestamps to include in evaluation (used by dataset loader). If None,
            treated as very large. Default is 1000.

        to_cache : bool, optional
            If True, stores intermediate benchmark artifacts/results into cache (when supported by your pipeline),
            so repeated runs can be faster/reproducible. Default is True.

        use_cache : bool, optional
            If True, attempts to load cached intermediate artifacts/results instead of recomputing them.
            Default is True.

        fixed_rate : float, optional
            Fixed missingness rate used in contexts where a single rate is required (e.g., some per-class routines,
            baseline comparisons, or when overriding dynamic rate selection). Default is 0.2.
            If put to -1, the same rate as the x_axis will be used

        bypass_error : bool, optional
            If True, this variable will put automatically NaN values if an error is encountered.
            Put to true if you want to get the results of all success imputer

        generate_plot: bool, optional
            If True, generates and saves plots (e.g., heatmaps/line plots) for benchmark results.
            Default is True.

        plots : bool, optional
            If True, display plots (e.g., heatmaps/line plots) for benchmark results.
            Default is True.

        inner_plots : bool, optional
            If True, generates and saves every steps plots for each individual datasets for benchmark results.
            Default is True.

        verbose : bool, optional
            Verbosity flag controlling detailed logging and plotting behavior. Default is False.

        Example
        -------
        .. code-block:: python

            from imputegap.recovery.benchmark import Benchmark
            from imputegap.tools import utils


        """

        # =============================================================================================================
        # DATA AND CONFIGURATION MANAGEMENT
        # =============================================================================================================
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        import gc
        try:
            import torch
        except ImportError:
            torch = None

        if algorithms is None:
            algorithms = ["cdrec"]
        if datasets is None:
            datasets = ["chlorine"]
        if patterns is None:
            patterns = ["mcar"]
        if x_axis is None:
            x_axis = [0.1, 0.2, 0.4, 0.6, 0.8]
        if metrics is None:
            metrics = ["*"]

        run_storage = []

        if not isinstance(algorithms, list):
            raise TypeError(f"'algorithms' must be a list, but got {type(algorithms).__name__}")
        if not isinstance(datasets, list):
            raise TypeError(f"'datasets' must be a list, but got {type(datasets).__name__}")
        if not isinstance(patterns, list):
            raise TypeError(f"'patterns' must be a list, but got {type(patterns).__name__}")
        if not isinstance(x_axis, list):
            raise TypeError(f"'x_axis' must be a list, but got {type(x_axis).__name__}")

        if metrics is None or ("*" in metrics or "all" in metrics):
            metrics = utils.list_of_metrics()
        if "*" in algorithms or "all" in algorithms:
            algorithms = utils.list_of_algorithms()
        if "*" in datasets or "all" in datasets:
            datasets = utils.get_dataset_imputations(directory='datasets/')
        if "*" in patterns or "all" in patterns:
            patterns = ["mcar", "aligned_series", "aligned_timestamps"]
        if "*" in x_axis or "all" in x_axis:
            x_axis = [0.1, 0.2, 0.4, 0.6, 0.8]

        if fixed_rate <= 0:
            dynammic_fixed = True
        else:
            dynammic_fixed = False
        loaded_cache = False
        evaluate_upstream = True
        model = "imputation"

        directory_now = datetime.datetime.now()
        directory_time = directory_now.strftime("%y_%m_%d_%H_%M_%S")
        save_dir = save_dir + "/" + "benchmark_imputation_" + report_title + "_" + directory_time
        b_task = "default"

        if nbr_series is None:
            nbr_series = 10000000
        if nbr_vals is None:
            nbr_vals = 10000000
        if not generate_plot:
            plots = False
            inner_plots = False

        benchmark_time = time.time()
        runs=1

        definition_of_exp = f"\nThe benchmark has been called (imputation):\n\talgorithms: {algorithms}\n\tdatasets: {datasets}\n\tpatterns: {patterns}\n\tmissing_percentages: {x_axis}\n\tmetrics: {metrics}\n\tnormalizer: {normalizer}\n\tfixed_rate: {fixed_rate}\n\tto_cache: {to_cache}\n\tuse_cache: {use_cache}\n\tbypass_error: {bypass_error}\n\tnumber max series: {nbr_series}\n\tnumber max values: {nbr_vals}\n\n"
        print(definition_of_exp)

        if normalizer == "denorm":
            #denormalization = True
            normalizer = "z_score"
        #else:
            #denormalization = False

        # =============================================================================================================
        # BENCHMARK
        # =============================================================================================================
        for i_run in range(0, abs(runs)):
            for dataset in datasets:

                runs_plots_scores = {}
                block_size_mcar = 10
                y_p_size = max(4, len(algorithms) * 0.275)

                # -----------------------------------------------------------------------------------------------------
                # DATASET LOADING
                # -----------------------------------------------------------------------------------------------------
                ts = TimeSeries(verbose=verbose)
                ts.load_series(utils.search_path(dataset), normalizer=normalizer, verbose=verbose, nbr_series=nbr_series, nbr_val=nbr_vals)
                N, M = ts.data.shape
                print(f"{dataset}:{ts.data.shape = }")

                if M <= 0:
                    raise ValueError(f"The dataset loaded has no series (series {M}).")

                for pattern in patterns:  # for each pattern e.g, mcar

                    for algorithm in algorithms:  # for each algorithm e.g, cdrec

                        print(f"{algorithm} is tested with {pattern} on {dataset}, started at {time.strftime('%Y-%m-%d %H:%M:%S')}.")

                        #for entry in utils.list_of_algorithms_with_families():
                        #    family, name = entry.split(".", 1)
                        #    if name.casefold() == algorithm.casefold():
                        #        model=family.lower()

                        for incx, x in enumerate(x_axis):  # for each missingness rate e.g, 0.2

                            if dynammic_fixed:
                                fixed_rate = x

                            try:
                                start_time_imputation = time.time()

                                # ------------------------------------------------------------------------------------
                                # CACHING HANDLING
                                # ------------------------------------------------------------------------------------
                                if use_cache:
                                    name, dir_cache = utils.prepare_caching(dataset=dataset, algorithm=algorithm, pattern=pattern, x=x, contamination_by_class=None, imputation_by_class=None, fixed_rate=fixed_rate)

                                    name = "_imp_" + name
                                    cache_path_recov = os.path.join(dir_cache, name + "_recov.txt")
                                    cache_path_missing = os.path.join(dir_cache, name + "_miss.txt")
                                    ts_r = TimeSeries(verbose=False)
                                    ts_m = TimeSeries(verbose=False)

                                    if os.path.exists(cache_path_recov) and os.path.exists(cache_path_missing):
                                        ts_r.load_series(cache_path_recov, normalizer=None, verbose=False)
                                        ts_m.load_series(cache_path_missing, normalizer=None, verbose=False)
                                        imputer = utils.config_impute_algorithm(incomp_data=ts_m.data, algorithm=algorithm, verbose=verbose)
                                        imputer.recov_data = ts_r.data
                                        imputer.task = b_task
                                        loaded_cache = True
                                        print(f"\tthe imputed matrix (imputation) has been loaded from the cache {ts_r.data.shape}: {name}")
                                    else:
                                        loaded_cache = False

                                # ------------------------------------------------------------------------------------
                                # CONTAMINATION OF THE DATA
                                # ------------------------------------------------------------------------------------
                                if not loaded_cache and not bypass_error:
                                    if pattern == "aligned_series" or pattern == "seqn":
                                        incomp_data = utils.config_contamination(ts=ts, pattern="aligned", dataset_rate=fixed_rate, series_rate=x, verbose=verbose, offset=0.05)
                                    elif pattern == "aligned_timestamps" or pattern == "blks":
                                        incomp_data = utils.config_contamination(ts=ts, pattern="aligned", dataset_rate=x, series_rate=fixed_rate, verbose=verbose, offset=0.05)
                                    else:
                                        incomp_data = utils.config_contamination(ts=ts, pattern=pattern, dataset_rate=fixed_rate, series_rate=x, block_size=block_size_mcar, verbose=verbose, offset=0.05)
                                    miss = incomp_data.copy()

                                    try:
                                        # ------------------------------------------------------------------------------------
                                        # IMPUTATION OF THE DATA
                                        # ------------------------------------------------------------------------------------
                                        imputer = utils.config_impute_algorithm(incomp_data=miss, algorithm=algorithm, task=b_task, verbose=verbose)
                                        imputer.verbose = verbose
                                        imputer.task = b_task
                                        imputer.impute()

                                    except Exception as e:
                                        print(f"Error during imputation: {e}")

                                if (not loaded_cache and not bypass_error) or (loaded_cache and bypass_error) or (loaded_cache and not bypass_error):
                                    end_time_imputation = time.time()

                                    dataset_s = dataset
                                    if "-" in dataset:
                                        dataset_s = dataset.replace("-", "")
                                    save_dir_plot = save_dir + "/" + dataset_s + "/" + pattern + "/recovery/"
                                    cont_rate = int(x * 100)
                                    imputer.score(input_data=ts.data, recov_data=imputer.recov_data, verbose=False)

                                    # STORAGE OF THE CACHED RESULTS
                                    # -----------------------------------------------------------------------------
                                    if to_cache and not loaded_cache:
                                        if not np.isnan(imputer.recov_data).any() and imputer.recov_data.shape == miss.shape:
                                            name, dir_cache = utils.prepare_caching(dataset=dataset, algorithm=algorithm, pattern=pattern, x=x, contamination_by_class=None, imputation_by_class=None, fixed_rate=fixed_rate)
                                            name = "_imp_" + name
                                            utils.ts_caching_save(data=imputer.recov_data, type="recov", artifact=None, name=name, dir_cache=dir_cache)
                                            utils.ts_caching_save(data=miss, type="miss", artifact=None, name=name, dir_cache=dir_cache)
                                            print(f"\tcaching stored for recovery and missing matrix: {name}")
                                        else:
                                            print(f"\t\t\tcaching FAILED, the recovery matrix has NaNs on it...: {algorithm}")

                                    # RESULT AND METRICS
                                    # -----------------------------------------------------------------------------
                                    if "*" not in metrics and "all" not in metrics:
                                        imputer.metrics = {k: imputer.metrics[k] for k in metrics if k in imputer.metrics}

                                    time_imputation = (end_time_imputation - start_time_imputation) * 1000
                                    if time_imputation < 1:
                                        time_imputation = 1
                                    log_time_imputation = math.log10(
                                        time_imputation) if time_imputation > 0 else None

                                    imputer.metrics["RUNTIME"] = time_imputation
                                    imputer.metrics["RUNTIME_LOG"] = log_time_imputation
                                    imputer.metrics["DOWNSTREAM_SMAPE"] = np.nan

                                    if inner_plots:
                                        try:
                                            ts.plot(input_data=ts.data, incomp_data=imputer.incomp_data,
                                                    recov_data=imputer.recov_data, nbr_series=6, subplot=True,
                                                    algorithm=imputer.algorithm, cont_rate=str(cont_rate),
                                                    display=False, save_path=save_dir_plot, verbose=False)
                                        except Exception as e:
                                            print(f"Error during plotting: {e}")

                                    if evaluate_upstream:
                                        runs_plots_scores.setdefault(str(dataset_s), {}).setdefault(str(pattern), {}).setdefault(str(algorithm), {}).setdefault(str(model), {})[str(x)] = {"scores": imputer.metrics}

                                else:
                                    print(f"\t\t{bypass_error=} / {loaded_cache=}")
                                    imputer_tr, runs_plots_scores = self._benchmark_error("bypass", None, runs_plots_scores, ts_m, [model], dataset, algorithm, pattern, x, save_dir, verbose)

                                del imputer
                                gc.collect()
                                if torch is not None and torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    torch.cuda.ipc_collect()

                            except Exception as e:
                                import traceback

                                print(f"\nERROR: {type(e).__name__}: {e}")
                                traceback.print_exc()

                                ts_m = np.zeros(len(ts.data))

                                imputer_tr, runs_plots_scores = self._benchmark_error("error", e, runs_plots_scores, ts_m, [model], dataset, algorithm, pattern, x, save_dir, verbose)

                        print("done!\n\n")

                run_storage.append(runs_plots_scores)

        plt.close('all')  # Close all open figures

        # PLOTTING AND RESULTS
        # -----------------------------------------------------------------------------
        for x, m in enumerate(reversed(metrics)):
            scores_list, algos, sets = self.avg_results_task(*run_storage, metric=m, upstream=evaluate_upstream)
            try:
                if generate_plot:
                    _ = self.generate_heatmap(scores_list=scores_list, algos=algos, sets=sets, metric=m, save_dir=save_dir, display=False)
                if not plots:
                    plt.close('all')
            except Exception as e:
                print(f"Error during plotting: {e}")

        benchmark_end = time.time()
        total_time_benchmark = round(benchmark_end - benchmark_time, 4)
        h = int(total_time_benchmark // 3600)
        m = int((total_time_benchmark % 3600) // 60)
        s = total_time_benchmark % 60
        print(f"\n> logs: benchmark - Execution Time: {total_time_benchmark} seconds,  ({h:02d}:{m:02d}:{s:06.3f} hh:mm:ss)\n")

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "runtime.log")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(save_path, "a") as file:
            file.write(f"{timestamp} | logs: benchmark - Execution Time: {total_time_benchmark} seconds,  ({h:02d}:{m:02d}:{s:06.3f} hh:mm:ss)\n")
        verb = False

        for scores in run_storage:
            all_keys = list(scores.keys())
            dataset_name = str(all_keys[0])
            save_dir_agg_set = save_dir + "/" + dataset_name

            try:
                if generate_plot:
                    self.generate_reports_txt(runs_plots_scores=scores, save_dir=save_dir_agg_set, given_dataset=dataset_name, metrics=metrics, rt=total_time_benchmark, upstream=evaluate_upstream, run=-1)
            except Exception as e:
                print(f"Error during file report generation: {e}")
            try:
                if generate_plot:
                    self.generate_plots(runs_plots_scores=scores, ticks=x_axis, metrics=metrics, subplot=True, y_size=y_p_size, save_dir=save_dir_agg_set, upstream=evaluate_upstream, display=verb)
                if not plots:
                    plt.close('all')
            except Exception as e:
                print(f"Error during plotting: {e}")

        try:
            self.generate_reports_summary(run_of_values=run_storage, save_dir=save_dir, metrics=metrics, rt=total_time_benchmark, run=-1, title=report_title, upstream=evaluate_upstream)
        except Exception as e:
            print(f"Error during file report generation: {e}")

        try:
            self.save_into_excel(run_of_values=run_storage, xlsx_path=os.path.join(save_dir, f"_summary_{report_title}.xlsx"), upstream=evaluate_upstream, verbose=verbose)
        except Exception as e:
            print(f"Error during report generation: {e}")

        print(f"\nThe results are saved in : {save_dir}\n")
        self.list_results = run_storage

        if evaluate_upstream:
            self.aggregate_results = scores_list

        save_def = os.path.join(save_dir, "experimentation_setup.log")

        try:
            with open(save_def, "w") as file:
                file.write(definition_of_exp)
        except Exception as e:
            print(f"Error during report generation: {e}")