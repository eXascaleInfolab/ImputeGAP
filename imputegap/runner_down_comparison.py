# ImputeGAP imports
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.contamination import GenGAP
from imputegap.recovery.downstream import Downstream

# initialize the time series object
ts = TimeSeries()
print(f"\nImputeGAP downstream models for forcasting : {ts.forecasters}")

# load and normalize the dataset
ts.load_series(data="chlorine", normalizer="z_score")

# contaminate the time series
ts_m = GenGAP.mcar(input_data=ts.data, rate_series=0.4, offset=0)

# reverse the contamination to amplify its impact on forecasting.
ts_m = GenGAP.mirror_gap(input_data=ts.data, ts_m=ts_m)

# define the downstream options
downstream = Downstream(input_data=ts.data, incomp_data=ts_m)

# launch the analysis
comparison = downstream.comparator(imp_algs=["CDRec", "MICE", "ZeroImpute"],
                                   model="chronos",
                                   horizons=[0.36],
                                   season=24,
                                   meta_pattern="mcar",
                                   meta_rate=0.4)