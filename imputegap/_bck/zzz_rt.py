import io
import os

from imputegap.recovery.imputation import Imputation
from imputegap.recovery.contamination import GenGap
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.downstream import Forecaster
from imputegap.recovery.downstream import Artifact

from imputegap.tools import utils
import time

for x in [500, 1000, 5000, 10000, 15000]:

    for algos in ["MissNet", "GPT4TS", "Moment"]:

        try:
            # initialize the time series object
            ts = TimeSeries()
            print(f"\nImputeGAP downstream models for forcasting : {ts.forecasters}")

            artifact = Artifact(dataset="etth1f", model="exp-smoothing", algorithm=algos, pattern="mcar", horizon=12)

            # load and normalize the dataset
            ts.load_forecasting_dataset(artifact.dataset, normalizer="z-score", verbose=True, horizon=artifact.horizon, nbr_val=x+12)

            y_train, season = ts.load_proper_output_forecasting()

            # contaminate the time series
            ts_m = GenGap.mcar(ts.data, rate_dataset=0.2, rate_series=0.2, offset=0)
            ts_m = GenGap.mirror_gap(raw_data=ts.data, ts_m=ts_m)


            imputer = utils.config_impute_algorithm(incomp_data=ts_m, algorithm=algos, verbose=False)
            imputer.verbose = False
            start_time_imputation = time.time()
            imputer.impute()
            end_time_imputation = time.time()
            time_imputation = (end_time_imputation - start_time_imputation) * 1000

            imputer.score(input_data=ts.data, recov_data=imputer.recov_data)

            #write the run time in a txt file > x=____, algos=_____, rt=____________
            # name = {x}_{algos}_rt.txt
            # pos = ./_optimization/ > use io.
            # to do

            here = os.path.dirname(os.path.dirname(__file__))
            data = os.path.join(here, "imputegap/_optimization/_rt/")
            os.makedirs(data, exist_ok=True)

            filename = f"{x}_{algos}_rt.txt"
            filepath = os.path.join(data, filename)

            with io.open(filepath, "w", encoding="utf-8") as f:
                f.write(f"x={x}, algos={algos}, rt={time_imputation}, rmse={imputer.metrics["RMSE"]}\n")

            print(f"\nImputation {x}-{algos} : DONE\n\t{filepath=}")
            print(f"x={x}, algos={algos}, rt={time_imputation}, rmse={imputer.metrics["RMSE"]}\n")
        except Exception as e:
            print(f"\nImputation failed. {x}-{algos}\n{e}")
