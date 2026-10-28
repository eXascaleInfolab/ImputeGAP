# ImputeGAP imports
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.contamination import GenGap
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.downstream import Forecaster
from imputegap.recovery.downstream import Artifact


# initialize the time series object
ts = TimeSeries()
print(f"\nImputeGAP downstream models for forcasting : {ts.forecasters}")

artifact = Artifact(dataset="paris", model="exp-smoothing", algorithm="cdrec", pattern="mcar", horizon=0.4)

# load and normalize the dataset
ts.load_forecasting_dataset(artifact.dataset, normalizer="z-score", verbose=True, horizon=artifact.horizon)

y_train, season = ts.load_proper_output_forecasting()

# contaminate the time series
ts_m = GenGap.aligned(ts.data, rate_dataset=artifact.fixed_x, rate_series=artifact.x, offset=0)
ts_m = GenGap.mirror_gap(raw_data=ts.data, ts_m=ts_m)

# define and impute the contaminated series
imputer = Imputation.MatrixCompletion.CDRec(ts_m)
imputer.impute()

# compute and print the downstream results
forecaster = Forecaster(task="forecast",
                        model=artifact.model,
                        y_train=y_train,
                        season=season,
                        use_cache=False,
                        to_cache=True,
                        artifact=artifact,
                        plots=True)

imputer.score(input_data=ts.data, recov_data=imputer.recov_data)
imputer.score(input_data=ts.data, recov_data=imputer.recov_data, downstream=forecaster)

ts.print_results(imputer.metrics, text="Upstream results")
ts.print_results(imputer.downstream_metrics, text="Downstream results")
