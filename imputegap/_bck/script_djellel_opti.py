import os
import numpy as np
import time
import pandas as pd

from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils
from recovery.contamination import GenGap


def optimize_adv_imputation_algo_classification(task="classification", datasets=["Beef"], normalizer="z-score", horizon=12, test_imputation=False, verbose=False):
    from imputegap.env.default_opti import OPTI_DICTIONARY
    from imputegap.env.default_opti import OPTI_TEST_DICTIONARY

    print(f"\t{OPTI_TEST_DICTIONARY=}")

    for key, entry in OPTI_TEST_DICTIONARY.items(): # loop on each algorithm to optimize
        algorithm, param_to_opti = key.split("-", 1)
        results = []

        print(f"{algorithm} | {param_to_opti}: {entry=} ")

        for value in entry: # add the value of the params
            params = {param_to_opti: value}
            print("\n")

            for dataset in datasets: # loop on the 80's datasets
                print(f"\t\t\t\topti: {dataset} - {algorithm} = {params=}")

                try:
                    ts = TimeSeries(verbose=False)
                    if task == "classification":
                        ts.load_classify_dataset(dataset, normalizer=normalizer, to_numpy=True, test_imputation=test_imputation, nbr_series=10000, nbr_val=1000, verbose=False)

                        classify_sets = ts.load_proper_output_classify()
                        X_test, y_train, y_test, _, _, test_imputation, _, dataset = classify_sets

                        start_time_imputation = time.time()

                        imputer, _, miss = utils.prepare_series_by_class(labels=y_train, raw_data=ts.data.copy(), imputer=algorithm, params=params, pattern="mcar", rate_dataset=0.2, rate_series=0.2, offset=0.05, verbose=False)

                        end_time_imputation = time.time()
                        time_imputation = (end_time_imputation - start_time_imputation) * 1000

                        imputer.score(input_data=ts.data, recov_data=imputer.recov_data, verbose=False)
                        imputer.metrics["RUNTIME"] = time_imputation
                    else:
                        ts.load_forecasting_dataset(dataset, normalizer=normalizer, verbose=verbose, horizon=horizon, nbr_series=10000, nbr_val=10000)
                        y_train, season = ts.load_proper_output_forecasting()
                        incomp_data = utils.config_contamination(ts=ts, pattern="mcar", dataset_rate=0.2, series_rate=0.2, verbose=False, offset=0)
                        miss = GenGap.mirror_gap(raw_data=ts.data, ts_m=incomp_data)
                        start_time_imputation = time.time()

                        imputer = utils.config_impute_algorithm(incomp_data=miss, algorithm=algorithm, verbose=verbose)
                        imputer.impute()

                        end_time_imputation = time.time()
                        time_imputation = (end_time_imputation - start_time_imputation) * 1000

                        imputer.score(input_data=ts.data, recov_data=imputer.recov_data, verbose=False)
                        imputer.metrics["RUNTIME"] = time_imputation


                    row = {"dataset": dataset, "algorithm": algorithm, param_to_opti: value, "rmse": imputer.metrics["RMSE"], "mae": imputer.metrics["MAE"], "mi": imputer.metrics["MI"], "correlation": imputer.metrics["CORRELATION"], "runtime": imputer.metrics["RUNTIME"], }
                    results.append(row)

                except:
                    print(f"\n\nERROR WITH OPTIMIZATION {algorithm=}, {params=} ! \n\n")
                    row = {"dataset": dataset, param_to_opti: value, "rmse": np.nan, "mae": np.nan, "mi": np.nan, "correlation": np.nan, "runtime": np.nan, }
                    results.append(row)

        try:
            here = os.path.dirname(os.path.abspath(__file__))
            opti_dir = os.path.join(here, "_optimization")
            os.makedirs(opti_dir, exist_ok=True)
            path_opti = os.path.join(opti_dir, f"{algorithm}_opti_results.xlsx")
            df = pd.DataFrame(results)

            try:
                summary = (df.groupby(param_to_opti, dropna=False).agg(avg_rmse=("rmse", "mean"), avg_runtime=("runtime", "mean"), avg_mae=("mae", "mean"), avg_mi=("mi", "mean"), avg_correlation=("correlation", "mean"), n_valid_rmse=("rmse", "count"), n_valid_runtime=("runtime", "count"), n_total=("dataset", "count"), ).reset_index())
                summary["rank_rmse"] = summary["avg_rmse"].rank(method="min", ascending=True, na_option="bottom" )
                summary["rank_runtime"] = summary["avg_runtime"].rank(method="min", ascending=True, na_option="bottom")
                df = df.merge(summary[[param_to_opti, "avg_rmse", "avg_runtime", "rank_rmse", "rank_runtime"]], on=param_to_opti, how="left" )
            except:
                df["avg_rmse"] = 0
                df["avg_runtime"] = 0
                df["rank_rmse"] = 0
                df["rank_runtime"] = 0
                print(f"cannot rank for {algorithm}...")

            df.to_excel(path_opti, index=False)
            print(f"results of {algorithm}, saved in {path_opti=}...")

        except:
            print(f"cannot save results of {algorithm}...")

#datasets = utils.get_datasets_classifiers()
#datasets=['Adiac', 'ArrowHead', 'Beef', 'BirdChicken', 'CBF', 'Car', 'CinCECGTorso', 'Colposcopy', 'Computers', 'CricketX', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'ECGFiveDays', 'EOGHorizontalSignal', 'EOGVerticalSignal', 'Earthquakes', 'EthanolLevel', 'FaceFour', 'FacesUCR', 'Fish', 'FreezerRegularTrain', 'FreezerSmallTrain', 'GunPoint', 'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'Ham', 'Haptics', 'Herring', 'InlineSkate', 'InsectEPGRegularTrain', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7', 'Meat', 'MedicalImages', 'MoteStrain', 'OSULeaf', 'OliveOil', 'PhalangesOutlinesCorrect', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxOutlineCorrect', 'RefrigerationDevices', 'Rock', 'ScreenType', 'SemgHandGenderCh2', 'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShapeletSim', 'SharePriceIncrease', 'SmallKitchenAppliances', 'SonyAIBORobotSurface1', 'Strawberry', 'SwedishLeaf', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'TwoPatterns', 'UMD', 'Wafer', 'Wine', 'Worms', 'WormsTwoClass', 'Yoga']
task_cla, datasets_cla = "classification", ['Car']
task_for, datasets_for = "forecasting", ['paris']
optimize_adv_imputation_algo_classification(task=task_for, datasets=datasets_for)