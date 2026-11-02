# ImputeGAP imports
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.contamination import GenGap
from imputegap.recovery.downstream import Downstream
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the time series object
ts = TimeSeries()
print(f"\nImputation algorithms : {ts.algorithms}")

missing_rate, missingness_pattern = 0.2, "mcar"

# load and normalize the dataset
ts.load_series(utils.search_path("chlorine"), normalizer="z_score")

# contaminate the time series
ts_m = GenGap.mcar(ts.data, rate_dataset=missing_rate, rate_series=missing_rate)

# impute the contaminated series
imputer = Imputation.MatrixCompletion.CDRec(ts_m)
imputer.impute()

# compute and print the imputation metrics
imputer.score(ts.data, imputer.recov_data)
ts.print_results(imputer.metrics)

# Downstream Comparator
downstream = Downstream()
down_results = downstream.comparator(recov_data=imputer.recov_data,
                                     input_data=ts.data,
                                     incomp_data=imputer.incomp_data,
                                     meta_algorithm="cdrec",
                                     horizons=[24],
                                     models=["chronos", "patchtst", "hw-add", "dlinear"],
                                     meta_pattern=missingness_pattern,
                                     meta_rate=missing_rate)