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
ts_m = GenGAP.mcar(input_data=ts.data, rate_series=0.2, offset=0, reverse=True)

# define and impute the contaminated series
imputer = Imputation.MatrixCompletion.CDRec(incomp_data=ts_m)
imputer.impute()

# define the parameters for the downstream model
forecaster = Forecaster(artifact=Artifact(model="chronos", horizon=128, season=24), pipeline=True)

# compute the downstream analysis and display the results
imputer.score(input_data=ts.data, recov_data=imputer.recov_data, downstream=forecaster)