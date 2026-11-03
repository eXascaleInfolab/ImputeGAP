# ImputeGAP imports
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.contamination import GenGap
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils
from imputegap.recovery.downstream import Artifact, Forecaster

# initialize the time series object
ts = TimeSeries()
print(f"\nImputeGAP downstream models for forcasting : {ts.forecasters}")

# load and normalize the dataset
ts.load_series(utils.search_path("chlorine"), normalizer="z_score")

# contaminate the time series
ts_m = GenGap.mcar(ts.data, rate_series=0.4, offset=0)

# define and impute the contaminated series
imputer = Imputation.MatrixCompletion.CDRec(ts_m)
imputer.impute()

# define downstream model
forecaster = Forecaster(artifact=Artifact(model="chronos", horizon=0.1, season=24), pipeline=True)

imputer.score(ts.data, imputer.recov_data, downstream=forecaster)
ts.print_results(imputer.downstream_metrics, text="Downstream results")