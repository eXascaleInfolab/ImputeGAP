# ImputeGAP imports
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the time series object
ts = TimeSeries()
print(f"\nImputation algorithms : {ts.algorithms}")

# load the dataset that contains nans values
ts.load_series(data="test-logic-nans-llm", normalizer=None)

# skip the contamination and impute the series
imputer = Imputation.MatrixCompletion.CDRec(incomp_data=ts.data, use_cuda=0)
imputer.impute()
print(f"Reconstructed data:\n{imputer.recov_data[:10]}")

# save the imputed data
path = utils.ts_caching_save(data=imputer.recov_data, type="imputation", dir_cache="./_caching")
print(f"\nReconstruction saved: {path}")
