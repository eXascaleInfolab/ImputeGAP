from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.contamination import GenGap
from imputegap.tools import utils

# initialize the time series object
ts = TimeSeries()
print(f"\nMissingness patterns : {ts.patterns}")

# load and normalize the dataset
ts.load_series(utils.search_path("eeg-alcohol"), normalizer="z_score")

# contaminate the time series with MCAR pattern
ts_m = GenGap.mcar(ts.data, rate_dataset=0.2, rate_series=0.4, block_size=10, seed=True)

# plot the contaminated time series
ts.plot(ts.data, ts_m, nbr_series=9, subplot=False, save_path="./imputegap_assets/contamination")