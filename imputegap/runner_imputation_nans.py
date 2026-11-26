# ImputeGAP imports
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the time series object
ts = TimeSeries()
print(f"\nImputation algorithms : {ts.algorithms}")

# load the dataset that contains nans values
ts.load_series(data="chlorine_miss.txt", normalizer=None)
print(f"\nSample data:\n{ts.data[:30, 0]}\n")

# skip the contamination and impute the series
imputer = Imputation.MatrixCompletion.CDRec(incomp_data=ts.data, use_cuda=0)
imputer.impute()
print(f"\nReconstructed data:\n{imputer.recov_data[:30, 0]}\n")

# save the imputed data
path = utils.ts_caching_save(data=imputer.recov_data, type="imputation", dir_cache="./_caching")
print(f"\nReconstruction saved: {path}")
