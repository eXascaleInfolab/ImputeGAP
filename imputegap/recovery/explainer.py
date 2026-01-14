import math
import os
import time
import importlib.resources

import numpy as np
import pandas as pd
import toml
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils


class Explainer:
    """
    A class to manage SHAP-based model explanations and feature extraction for time series datasets.

    Methods
    -------
    show()
        Show the shap result plot

    load_configuration(file_path=None)
        Load categories and features from a TOML file.

    extractor_pycatch(data, features_categories, features_list, do_catch24=True)
        Extract features from time series data using pycatch22.

    extractor_tsfresh(data, categories=["statistical", "temporal", "frequency", "shape"])
        Extract features from time series data using TSFresh.
        The function supports filtering by feature categories and calculates a large set of features
        (up to 783) related to statistics, temporal dynamics, frequency analysis, and shape.

    extractor_tsfel(data, frequency=None, categories=["spectral", "statistical", "temporal", "fractal"])
        Extract features from time series data using TSFEL (Time Series Feature Extraction Library).
        This method calculates features based on the selected categories and optionally uses the
        sampling frequency to compute frequency-domain features.

    print(shap_values, shap_details=None)
        Print SHAP values and details for display.

    convert_results(tmp, file, algo, descriptions, features, categories, mean_features, to_save)
        Convert SHAP raw results into a refined format for display.

    execute_shap_model(x_dataset, x_information, y_dataset, file, algorithm, splitter=10, display=False, verbose=False)
        Launch the SHAP model to explain the dataset features.

    shap_explainer(input_data, algorithm="cdrec", params=None, extractor="pycatch", incomp_data="mcar",
                   missing_rate=0.4, block_size=10, offset=0.1, seed=True, limitation=15, splitter=0,
                   file_name="ts", display=False, verbose=False)
        Handle parameters and set variables to launch the SHAP model.

    """

    def __init__(self):
        """
        Initialize the Explainer object.
        """
        self.shap_values = None
        self.shap_details = None
        self.plots = None

    def show(self):
        """
        Display the explainer plot for shap
        """
        import shap

        if self.plots is not None:
            plt.close("all")
            shval, x_test, optimal_display = self.plots
            shap.summary_plot(shval, x_test, plot_size=(25, 10), feature_names=optimal_display)
        else:
            print("(EXP) No plots saved in the object.")



    def load_configuration(self, file_path=None):
        """
        Load categories and features from a TOML file.

        Parameters
        ----------
        file_path : str, optional
            The path to the TOML file (default is None). If None, it loads the default configuration file.

        Returns
        -------
        tuple
            A tuple containing two dictionaries: categories, features and config.
        """

        if file_path is None:
            path = importlib.resources.files('imputegap.env').joinpath("./default_explainer.toml")
        else:
            if not os.path.exists(file_path):
                path = file_path[1:]

        config_data = toml.load(path)

        # Extract categories and features from the TOML data
        categories = config_data.get('CATEGORIES', {})
        features = config_data.get('FEATURES', {})

        return categories, features, config_data


    def extractor_tsfresh(self, data, categories=["statistical", "temporal", "shape", "frequency"]):
        """
        Extract features using tsfresh and group them into 4 categories:
        statistical, temporal, frequency, and shape-based.

        Parameters
        ----------
            data (numpy.ndarray):
                2D array of shape (M, N), where M is the timestamp and N is the number of series

            categories (list):
                List of categories to extract. Must include one or more of: ["statistical", "temporal", "frequency", "shape"]

        Returns
        -------
            dict: A dictionary with feature names as keys and their aggregated values as values.
            list: A list of tuples (feature_name, category, formatted_feature_name).
        """
        import tsfresh

        M, N = data.shape
        data = [[0 if num is None else num for num in sublist] for sublist in data]
        data = [[0 if num is None or (isinstance(num, (float, np.float32, np.float64)) and np.isnan(num)) else num for num in sublist] for sublist in data]
        data = np.array(data)

        results, descriptions = {}, []

        indices = np.tile(np.arange(N), M)  # Repeats 0,1,2,... for each series
        series_ids = np.repeat(np.arange(M), N)  # Assigns unique ID to each series
        values = data.flatten()  # Convert 2D array to 1D for tsfresh

        # Convert to a minimal DataFrame for tsfresh processing
        df = pd.DataFrame({"id": series_ids, "index": indices, "value": values})  # 'id' groups each series

        # Extract features (M, 783)
        features = tsfresh.extract_features(df, column_id="id", column_sort="index")

        # Aggregate for the whole dataset (1, 783)
        features_list = features.mean(axis=0).to_frame().T

        # Define the mapping of categories to keywords
        category_keywords = {
            categories[0]: {"mean", "variance", "standard_deviation", "sum", "median", "skewness", "kurtosis", "entropy"},
            categories[1]: {"autocorrelation", "change", "linear_trend", "time_reversal_asymmetry_statistic", "c3", "cid_ce", "symmetry", "number_crossing_m"},
            categories[2]: {"fft", "frequency", "cwt", "spkt_welch_density", "ar_coefficient", "energy_ratio", "permutation_entropy", "fourier_entropy"},
            categories[3]: {"peaks", "strike", "location", "range_count", "value_count", "large_standard_deviation", "ratio_beyond_r_sigma", "lempel_ziv_complexity"}
        }

        for inc, (feature_name, value) in enumerate(features_list.iloc[0].items()):
            if np.isnan(value):
                results[feature_name] = 0
            else:
                results[feature_name] = value

            for cat, keywords in category_keywords.items():
                if any(keyword in feature_name for keyword in keywords):
                    category = cat
                    break
            if category is None:
                print("\n\n\t\t\tOTHER CATEGORY DETECTED...", feature_name)
                category = categories[0]

            feature_name_desc = feature_name.replace("value__", "")
            descriptions.append((feature_name, category, feature_name_desc.replace("_", " ").title()))

        print(f"\ttsfresh : features extracted successfully___{inc} features")

        return results, descriptions




    def extractor_tsfel(self, data, frequency=None, categories=["spectral", "statistical", "temporal"]):
        """
        Extract features using TSFEL (Time Series Feature Extraction Library).

        This function extracts features from the input time series data based on the specified
        categories. The categories determine the type of features to compute, such as spectral,
        statistical, temporal, or fractal features. Optionally, a frequency value can be provided
        to compute frequency-specific features.

        Parameters
        ----------
            data (numpy.ndarray):
                2D array of shape (M, N), where M is the timestamp and N is the number of series

            frequency (float, optional):
                The sampling frequency of the time series data. This is used for spectral feature
                calculations (e.g., FFT-based features). If None, spectral features will be computed
                using default assumptions.

            categories (list, optional):
                A list of categories to extract. Valid categories are:
                    - "spectral": Extract frequency-domain features (e.g., FFT, spectral entropy).
                    - "statistical": Extract basic statistical features (e.g., mean, variance, skewness).
                    - "temporal": Extract temporal-domain features (e.g., autocorrelation, zero crossings).
                By default, all four categories are extracted.

        Returns
        -------
            dict:
                A dictionary where keys are feature names and values are the computed feature values
                for the entire dataset (aggregated over all time series).
            list:
                A list of tuples, where each tuple contains:
                    - Feature name (str): The name of the feature.
                    - Category (str): The category to which the feature belongs.
                    - Formatted feature name (str): A human-readable version of the feature name.

        Example:
            >>> import numpy as np
            >>> data = np.random.rand(5, 100)  # 5 time series, each with 100 values
            >>> results, descriptions = extractor_tsfel(data, frequency=50, categories=["statistical", "temporal"])
            >>> print(results)
            >>> print(descriptions)

        Notes:
            - This function requires TSFEL to be installed: `pip install tsfel`.
            - Categories can be customized to extract only the desired features, reducing computation time.
        """
        import tsfel

        data = [[0 if num is None else num for num in sublist] for sublist in data]
        data = [[0 if num is None or (isinstance(num, (float, np.float32, np.float64)) and np.isnan(num)) else num for num in sublist] for sublist in data]
        data = np.array(data)

        results, descriptions = {}, []

        import warnings
        warnings.filterwarnings("ignore")

        total_inc = 0
        for category in categories:
            # the spectral configuration
            # Extract features with TSFEL
            cfg = tsfel.get_features_by_domain(category)

            if category == "fractal":
                cfg = {name: params for name, params in cfg.items() if "hurst" not in name.lower()}

            features = tsfel.time_series_features_extractor(cfg, timeseries=data, fs=frequency)

            # Extract feature types by removing the ID prefix
            features.columns = features.columns.str.split('_', n=1).str[1]

            # Group by feature type to handle duplicates and compute the mean
            aggregated_features = features.groupby(features.columns, axis=1).mean()

            # Convert to a single-row DataFrame
            features_list = pd.DataFrame([aggregated_features.mean(axis=0)])

            # Print the shape and aggregated features
            for inc, (feature_name, value) in enumerate(features_list.iloc[0].items()):
                if np.isnan(value):
                    results[feature_name] = 0
                else:
                    results[feature_name] = value
                descriptions.append((feature_name, category, feature_name.replace("_", " ").title()))
            total_inc = total_inc + inc

        print(f"\ttsfel : features extracted successfully___{total_inc} features")

        return results, descriptions


    def extractor_pycatch(self, data, features_categories, features_list, do_catch24=True, deep_verbose=False):
        """
        Extract features from time series data using pycatch22.

        Parameters
        ----------
        data : numpy.ndarray
            Time series dataset for feature extraction.

        features_categories : dict
            Dictionary that maps feature names to categories.

        features_list : dict
            Dictionary of all features expected.

        do_catch24 : bool, optional
            Flag to compute the mean and standard deviation for Catch24 (default is True).

        deep_verbose : bool, optional
            Print the detail information of the extractor

        Returns
        -------
        tuple
            A tuple containing:
            - results (dict): A dictionary of feature values by feature names.
            - descriptions (list): A list of tuples containing feature names, categories, and descriptions.
        """
        import pycatch22

        data = [[0 if num is None else num for num in sublist] for sublist in data]
        data = [[0 if num is None or (isinstance(num, (float, np.float32, np.float64)) and np.isnan(num)) else num for num in sublist] for sublist in data]
        data = np.array(data)

        if isinstance(data, np.ndarray):
            flat_data = data.flatten().tolist()
        else:
            flat_data = [float(item) for sublist in data for item in sublist]

        if isinstance(flat_data[0], list):
            flat_data = [float(item) for sublist in flat_data for item in sublist]

        catch_out = pycatch22.catch22_all(flat_data, catch24=do_catch24)

        feature_names = catch_out['names']
        feature_values = catch_out['values']
        results, descriptions = {}, []

        if any(isinstance(value, (float, np.float32, np.float64)) and np.isnan(value) for value in feature_values):
            raise ValueError("Error: NaN value detected in feature_values")

        inc = 0
        for feature_name, feature_value in zip(feature_names, feature_values):

            print(f"{feature_name = } ({inc}/{len(feature_names)}) : {feature_value = }")

            results[feature_name] = feature_value

            for category, features in features_categories.items():
                if feature_name in features:
                    category_value = category
                    break

            feature_description = features_list.get(feature_name)

            descriptions.append((feature_name, category_value, feature_description))
            inc = inc + 1

        print(f"\tpycatch22 : features extracted successfully___{inc} features")

        return results, descriptions

    def print(self, shap_values, shap_details=None):
        """
        Convert SHAP raw results to a refined format for display.

        Parameters
        ----------
        shap_values : list
            The SHAP values and results of the SHAP analysis.
        shap_details : list, optional
            Input and output data of the regression, if available (default is None).

        Returns
        -------
        None
        """
        #output = ", ".join([f"{output}" for _, output in shap_details])
        #print(f"RMSE RESULTS (Y_TRAIN & Y_TEST): [{output}]")

        print("\nTop-5 features:")
        inc = 0
        for (x, algo, rate, description, feature, category, mean_features) in shap_values:
            inc = inc + 1
            print(f"\tFeature : {x:<5} {algo:<10} with a score of {rate:<10} {category:<18} {description:<75} {feature}\n")

            if inc > 5:
                break

    def convert_results(self, tmp, file, algo, descriptions, features, categories, mean_features, to_save):
        """
        Convert SHAP raw results to a refined format for display.

        Parameters
        ----------
        tmp : list
            Current SHAP results.
        file : str
            Dataset used.
        algo : str
            Algorithm used for imputation.
        descriptions : list
            Descriptions of each feature.
        features : list
            Raw names of each feature.
        categories : list
            Categories of each feature.
        mean_features : list
            Mean values of each feature.
        to_save : str
            Path to save results.

        Returns
        -------
        list
            A list of processed SHAP results.
        """

        result_display, result_shap = [], []
        for x, rate in enumerate(tmp):
            if not math.isnan(rate):
                rate = float(round(rate, 2))

            result_display.append(
                (x, algo, rate, descriptions[0][x], features[0][x], categories[0][x], mean_features[x]))

        result_display = sorted(result_display, key=lambda tup: (tup[1], tup[2]), reverse=True)

        with open(to_save + "_values.txt", 'w') as file_output:
            for (x, algo, rate, description, feature, category, mean_features) in result_display:
                file_output.write(
                    f"Feature : {x:<5} {algo:<10} with a score of {rate:<10} {category:<18} {description:<65} {feature}\n")
                result_shap.append([file, algo, rate, description, feature, category, mean_features])

        return result_shap

    def execute_shap_model(self, x_dataset, x_information, y_dataset, file, algorithm, splitter=10, extractor="pycatch", display=False, verbose=False):
        """
        Launch the SHAP model for explaining the features of the dataset.

        Parameters
        ----------
        x_dataset : numpy.ndarray
            Dataset of feature extraction with descriptions.
        x_information : list
            Descriptions of all features grouped by categories.
        y_dataset : numpy.ndarray
            RMSE labels of each series.
        file : str
            Dataset used for SHAP analysis.
        algorithm : str
            Algorithm used for imputation (e.g., 'cdrec', 'stmvl', 'iim', 'mrnn').
        splitter : int, optional
            Split ratio for data training and testing (default is 10).
        extractor : str
            Feature extractor used for the regression (e.g., 'pycatch', 'tsfel').
        display : bool, optional
            Whether to display the SHAP plots (default is False).
        verbose : bool, optional
            Whether to print detailed output (default is False).

        Returns
        -------
        list
            Results of the SHAP explainer model.
        """
        import shap

        print("\n\nInitialization of the SHAP model with dimension", np.array(x_information).shape)
        _, _, config = self.load_configuration()

        plots_categories = config[extractor]['categories']

        path_file = "./imputegap_assets/shap/"
        path_file_details = "./imputegap_assets/shap/analysis_grouped/"
        path_file_categories = "./imputegap_assets/shap/analysis_per_cat/"

        os.makedirs(path_file, exist_ok=True)
        os.makedirs(path_file_details, exist_ok=True)
        os.makedirs(path_file_categories, exist_ok=True)

        x_features, x_categories, x_descriptions = [], [], []
        x_fs, x_cs, x_ds, alphas = [], [], [], []

        for current_time_series in x_information:
            x_fs.clear()
            x_cs.clear()
            x_ds.clear()
            for feature_name, category_value, feature_description in current_time_series:
                x_fs.append(feature_name)
                x_cs.append(category_value)
                x_ds.append(feature_description)
            x_features.append(x_fs)
            x_categories.append(x_cs)
            x_descriptions.append(x_ds)

        x_dataset = np.array(x_dataset)
        y_dataset = np.array(y_dataset)


        x_dataset = np.nan_to_num(x_dataset, nan=0.0, posinf=10000, neginf=-10000)
        x_dataset = np.clip(x_dataset, -10000, 10000)

        x_features = np.array(x_features)
        x_categories = np.array(x_categories)
        x_descriptions = np.array(x_descriptions)

        # Split the data
        x_train, x_test = x_dataset[:splitter], x_dataset[splitter:]
        y_train, y_test = y_dataset[:splitter], y_dataset[splitter:]

        # Print shapes to verify
        print("\t SHAP_MODEL >> x_train shape:", x_train.shape)
        print("\t SHAP_MODEL >> y_train shape:", y_train.shape)
        print("\t SHAP_MODEL >> x_test shape:", x_test.shape)
        print("\t SHAP_MODEL >> y_test shape:", y_test.shape, "\n")
        if verbose:
            print("\t SHAP_MODEL >> extractor:", extractor)
            print("\t SHAP_MODEL >> features shape:", x_features.shape)
            print("\t SHAP_MODEL >> categories shape:", x_categories.shape)
            print("\t SHAP_MODEL >> descriptions shape:", x_descriptions.shape, "\n")
            print("\t SHAP_MODEL >> features OK:", np.all(np.all(x_features == x_features[0, :], axis=1)))
            print("\t SHAP_MODEL >> categories OK:", np.all(np.all(x_categories == x_categories[0, :], axis=1)))
            print("\t SHAP_MODEL >> descriptions OK:", np.all(np.all(x_descriptions == x_descriptions[0, :], axis=1)), "\n\n")

        model = RandomForestRegressor()

        model.fit(x_train, y_train)

        exp = shap.KernelExplainer(model.predict, x_test)
        shval = exp.shap_values(x_test)
        shval_x = exp(x_train)

        optimal_display = []
        for desc, group in zip(x_descriptions[0], x_categories[0]):
            optimal_display.append(desc + " (" + group + ")")

        series_names = []
        for names in range(0, np.array(x_test).shape[0]):
            series_names.append("Series " + str(names + np.array(x_train).shape[0]))

        self.plots = (shval, x_test, optimal_display)

        shap.summary_plot(shval, x_test, plot_size=(25, 10), feature_names=optimal_display, show=display)
        alpha = os.path.join(path_file + file + "_" + algorithm + "_" + extractor + "_shap_all.png")
        plt.title("SHAP Details Results")
        os.makedirs(path_file, exist_ok=True)
        plt.savefig(alpha)
        plt.close()
        alphas.append(alpha)


        if not display:
            shap.plots.waterfall(shval_x[0], show=display)
            alpha = os.path.join(path_file_details + file + "_" + algorithm + "_" + extractor + "_DTL_Waterfall.png")
            plt.title("SHAP Waterfall Results")
            fig = plt.gcf()  # Get the current figure created by SHAP
            fig.set_size_inches(20, 10)  # Ensure the size is correct
            plt.savefig(alpha)
            plt.close()
            alphas.append(alpha)

            shap.summary_plot(np.array(shval).T, np.array(x_test).T, feature_names=series_names, show=display)
            alpha = os.path.join(path_file_details + file + "_" + algorithm + "_" + extractor + "_shap_reverse.png")
            plt.title("SHAP Features by Series")
            plt.savefig(alpha)
            plt.close()
            alphas.append(alpha)

            shap.plots.beeswarm(shval_x, show=display, plot_size=(22, 10))
            alpha = os.path.join(path_file_details + file + "_" + algorithm + "_" + extractor + "_DTL_Beeswarm.png")
            plt.title("SHAP Beeswarm Results")
            plt.savefig(alpha)
            plt.close()
            alphas.append(alpha)

        total_weights_for_all_algorithms = []

        t_shval = np.array(shval).T
        t_Xtest = np.array(x_test).T

        aggregation_features, aggregation_test = [], []

        geometry, correlation, transformation, trend = [], [], [], []
        geometryDesc, correlationDesc, transformationDesc, trendDesc = [], [], [], []

        for index, feat in enumerate(t_shval):
            if x_categories[0][index] == plots_categories[0]:
                geometry.append(feat)
                geometryDesc.append(x_descriptions[0][index])
            elif x_categories[0][index] == plots_categories[1]:
                correlation.append(feat)
                correlationDesc.append(x_descriptions[0][index])
            elif x_categories[0][index] == plots_categories[2]:
                transformation.append(feat)
                transformationDesc.append(x_descriptions[0][index])
            elif x_categories[0][index] == plots_categories[3]:
                trend.append(feat)
                trendDesc.append(x_descriptions[0][index])

        geometryT, correlationT, transformationT, trendT = [], [], [], []
        for index, feat in enumerate(t_Xtest):
            if x_categories[0][index] == plots_categories[0]:
                geometryT.append(feat)
            elif x_categories[0][index] == plots_categories[1]:
                correlationT.append(feat)
            elif x_categories[0][index] == plots_categories[2]:
                transformationT.append(feat)
            elif x_categories[0][index] == plots_categories[3]:
                trendT.append(feat)

        mean_features = []
        for feat in t_Xtest:
            mean_features.append(np.mean(feat, axis=0))

        geometry = np.array(geometry)
        correlation = np.array(correlation)
        transformation = np.array(transformation)
        trend = np.array(trend)
        geometryT = np.array(geometryT)
        correlationT = np.array(correlationT)
        transformationT = np.array(transformationT)
        trendT = np.array(trendT)
        mean_features = np.array(mean_features)

        if not display:
            shap.summary_plot(np.array(geometry).T, np.array(geometryT).T, plot_size=(20, 10), feature_names=geometryDesc, show=display)
            alpha = os.path.join(path_file_categories + file + "_" + algorithm + "_" + extractor + "_shap_" + plots_categories[0].lower() + ".png")
            plt.title("SHAP details of " + plots_categories[0].lower())
            plt.savefig(alpha)
            plt.close()
            alphas.append(alpha)

            if len(plots_categories) >= 2 and plots_categories[1] != "":
                shap.summary_plot(np.array(transformation).T, np.array(transformationT).T, plot_size=(20, 10), feature_names=transformationDesc, show=display)
                alpha = os.path.join(path_file_categories + file + "_" + algorithm + "_" + extractor + "_shap_" + plots_categories[2].lower() + ".png")
                plt.title("SHAP details of " + plots_categories[1].lower())
                plt.savefig(alpha)
                plt.close()
                alphas.append(alpha)

            if len(plots_categories) >= 3 and plots_categories[2] != "":
                shap.summary_plot(np.array(correlation).T, np.array(correlationT).T, plot_size=(20, 10), feature_names=correlationDesc, show=display)
                alpha = os.path.join(path_file_categories + file + "_" + algorithm + "_" + extractor + "_shap_" + plots_categories[1].lower() + ".png")
                plt.title("SHAP details of " + plots_categories[1].lower())
                plt.savefig(alpha)
                plt.close()
                alphas.append(alpha)

            if len(plots_categories) >= 4 and plots_categories[3] != "":
                shap.summary_plot(np.array(trend).T, np.array(trendT).T, plot_size=(20, 8), feature_names=trendDesc, show=display)
                alpha = os.path.join(path_file_categories + file + "_" + algorithm + "_" + extractor + "_shap_" + plots_categories[3].lower() + ".png")
                plt.title("SHAP details of " + plots_categories[3].lower())
                plt.savefig(alpha)
                plt.close()
                alphas.append(alpha)

        for arr in (geometry, correlation, transformation, trend):
            if arr is not None and arr.size > 0:
                aggregation_features.append(np.mean(arr, axis=0))

        for arr in (geometryT, correlationT, transformationT, trendT):
            if arr is not None and arr.size > 0:
                aggregation_test.append(np.mean(arr, axis=0))

        aggregation_features = np.array(aggregation_features).T
        aggregation_test = np.array(aggregation_test).T

        shap.summary_plot(aggregation_features, aggregation_test, feature_names=plots_categories, show=display)
        alpha = os.path.join(path_file + file + "_" + algorithm + "_" + extractor + "_shap_cat.png")
        plt.title("SHAP Aggregation Results")
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.savefig(alpha)
        plt.close()
        alphas.append(alpha)

        if not display:
            shap.summary_plot(np.array(aggregation_features).T, np.array(aggregation_test).T, feature_names=series_names, show=display)
            alpha = os.path.join(path_file_details + file + "_" + algorithm + "_" + extractor + "_shap_agg_reverse.png")
            plt.title("SHAP Aggregation Features by Series")
            plt.savefig(alpha)
            plt.close()
            alphas.append(alpha)

        if verbose:
            print("\t\tSHAP Families details :")
            print("\t\t\tgeometry:", geometry.shape)
            print("\t\t\ttransformation:", transformation.shape)
            print("\t\t\tcorrelation:", correlation.shape)
            print("\t\t\ttrend':", trend.shape)
            print("\t\t\tmean_features:", mean_features.shape, "\n\n")

        # Aggregate shapely values per element of X_test
        total_weights = [np.abs(shval.T[i]).mean(0) for i in range(len(shval[0]))]

        # Convert to percentages
        total_sum = np.sum(total_weights)
        total_weights_percent = [(weight / total_sum * 100) for weight in total_weights]

        total_weights_for_all_algorithms = np.append(total_weights_for_all_algorithms, total_weights_percent)

        for alpha in alphas:
            print("\n\n\tplot has been saved : ", alpha)

        results_shap = self.convert_results(total_weights_for_all_algorithms, file, algorithm, x_descriptions,
                                                 x_features, x_categories, mean_features,
                                                 to_save=path_file + file + "_" + algorithm + "_" + extractor)

        return results_shap

    def shap_explainer(self, input_data, algorithm="cdrec", params=None, extractor="pycatch", pattern="mcar", missing_rate=0.4, block_size=10, offset=0.1, seed=True, rate_dataset=1, training_ratio=0.6, file_name="ts", display=False, verbose=False, deep_verbose=False):
        """
        Handle parameters and set variables to launch the SHAP model.

        Parameters
        ----------
        input_data : numpy.ndarray
            The original time series dataset.

        algorithm : str, optional
            The algorithm used for imputation (default is 'cdrec'). Valid values: 'cdrec', 'stmvl', 'iim', 'mrnn'.

        params : dict, optional
            Parameters for the algorithm.

        pattern : str, optional
            Contamination pattern to apply (default is 'mcar').

        extractor : str, optional
            Extractor use to get the features of the data (default is 'pycatch').  Valid values: 'pycatch', 'tsfel', 'tsfresh'

        missing_rate : float, optional
            Percentage of missing values per series (default is 0.4).

        block_size : int, optional
            Size of the block to remove at each random position selected (default is 10).

        offset : float, optional
            Size of the uncontaminated section at the beginning of the time series (default is 0.1).

        seed : bool, optional
            Whether to use a seed for reproducibility (default is True).

        rate_dataset : flaot, optional
            Limitation on the number of series for the model (default is 1).

        training_ratio : flaot, optional
            Limitation on the training series for the model (default is 0.6).

        file_name : str, optional
            Name of the dataset file (default is 'ts').

        display : bool, optional
            Whether to display the SHAP plots (default is False).

        verbose : bool, optional
            Whether to print detailed output (default is False).

        deep_verbose : bool, optional
            Whether to print debug output (default is False).

        Returns
        -------
        tuple
            A tuple containing:

            - shap_values : list
                SHAP values for each series.
            - shap_details : list
                Detailed SHAP analysis results.

        Notes
        -----
        The contamination is applied to each time series using the specified method. The SHAP model is then used
        to generate explanations for the imputation results, which are logged in a local directory.
        """
        start_time = time.time()  # Record start time

        if pattern in ["disjoint", "overlap", "blackout"]:
            raise ValueError("Invalid pattern detected: disjoint, overlap, or blackout are not allowed for SHAP.\nPlease, you MCAR, Aligned, Scattered, Gaussian, or Distribution.")

        if rate_dataset < 0.05 or rate_dataset > 1:
            print("\nlimit percentage higher than 100%, reduce to 100% of the dataset")
            rate_dataset = 1

        M = input_data.shape[1]
        limit = math.ceil(M * rate_dataset)

        if training_ratio < 0.05 or training_ratio > 0.95:
            print("\nsplit ratio to small or to high, reduce to 60% of the dataset")
            training_ratio = 0.6

        training_ratio = int(limit * training_ratio)

        if limit >= M:
            limit = M

        if verbose:
            print("\nFrom", limit, "/", M, "elements, the training dataset has been set with", training_ratio,"elements and the testing dataset with", (limit-training_ratio), "elements")

        print(f"\nexplainer launched"
              f"\n\textractor: {extractor}",
              f"\n\timputation algorithm: {algorithm}",
              f"\n\tparams: {params}",
              f"\n\tmissigness pattern: {pattern}"
              f"\n\tmissing rate: {missing_rate * 100}%"
              f"\n\tnbr of series training set: {training_ratio}"
              f"\n\tnbr of series testing set: {limit-training_ratio}")

        if extractor in ["pycatch22", "pycatch-22"]:
            extractor = "pycatch"

        input_data_matrices, obfuscated_matrices = [], []
        output_metrics, output_rmse, input_params, input_params_full = [], [], [], []

        if extractor == "pycatch" or extractor == "pycatch22" or extractor == "pycatch-22":
            categories, features, _ = self.load_configuration()

        for current_series in range(0, limit):

            print("\n\nGeneration ", current_series, "/", limit, "(", int((current_series / limit) * 100), "%)________________________________________________________")
            print("\tContamination ", current_series, "...")

            tmp = TimeSeries(verbose=False)
            tmp.import_matrix(input_data)
            incomp_data = utils.config_contamination(ts=tmp, pattern=pattern, dataset_rate=current_series, series_rate=missing_rate, block_size=block_size, offset=offset, seed=seed, explainer=True, verbose=False)

            input_data_matrices.append(input_data)
            obfuscated_matrices.append(incomp_data)

            if extractor == "pycatch":
                catch_fct, descriptions = self.extractor_pycatch(incomp_data, categories, features, False)
                extracted_features = np.array(list(catch_fct.values()))
            elif extractor == "tsfel":
                catch_fct, descriptions = self.extractor_tsfel(incomp_data)
                extracted_features = np.array(list(catch_fct.values()))
            elif extractor == "tsfresh":
                catch_fct, descriptions = self.extractor_tsfresh(incomp_data)
                extracted_features = np.array(list(catch_fct.values()))
            else:
                catch_fct, descriptions, extracted_features = None, None, None

            input_params.append(extracted_features)
            input_params_full.append(descriptions)

            print("\tImputation ", current_series, "...")
            algo = utils.config_impute_algorithm(incomp_data, algorithm, verbose=verbose)
            algo.logs = False
            algo.impute(user_def=True, params=params)
            algo.score(input_data)
            imputation_results = algo.metrics

            output_metrics.append(imputation_results)
            output_rmse.append(imputation_results["RMSE"])

        shap_details = []
        for input, output in zip(input_params, output_metrics):
            shap_details.append((input, output["RMSE"]))

        shap_values = self.execute_shap_model(input_params, input_params_full, output_rmse, file_name, algorithm, training_ratio, extractor, display, verbose)

        end_time = time.time()
        print(f"\n> logs: shap explainer - Execution Time: {(end_time - start_time):.4f} seconds\n")

        print("\nSHAP results saved in: ./imputegap_assets/shap/*\n")

        self.shap_values = shap_values
        self.shap_details = shap_details