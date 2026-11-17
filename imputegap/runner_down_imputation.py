# ImputeGAP imports
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.contamination import GenGAP
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.downstream import Artifact, Forecaster

# initialize the time series object
ts = TimeSeries()
print(f"\nImputeGAP downstream models for forcasting : {ts.forecasters}")

# load and normalize the dataset
ts.load_series(data="chlorine", normalizer="z_score")

# contaminate the time series
ts_m = GenGAP.mcar(input_data=ts.data, rate_series=0.2, offset=0)

# define and impute the contaminated series
imputer = Imputation.MatrixCompletion.CDRec(incomp_data=ts_m)

# reverse the contamination to amplify its impact on forecasting.
ts_m = GenGAP.mirror_gap(input_data=ts.data, ts_m=ts_m)

# define the parameters for the downstream model
forecaster = Forecaster(artifact=Artifact(model="chronos", horizon=0.36, season=24), pipeline=True)

# compute the downstream analysis and display the results
imputer.score(input_data=ts.data, recov_data=imputer.recov_data, downstream=forecaster)