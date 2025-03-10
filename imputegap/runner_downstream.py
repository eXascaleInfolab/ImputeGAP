from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# 1. initiate the TimeSeries() object that will stay with you throughout the analysis
ts_1 = TimeSeries()

# 2. load the timeseries from file or from the code
ts_1.load_series(utils.search_path("chlorine"))
ts_1.normalize(normalizer="min_max")

# 3. contamination of the data
ts_mask = ts_1.Contamination.missing_percentage(ts_1.data, rate_series=0.8)
ts_2 = TimeSeries().import_matrix(ts_mask)

# 4. imputation of the contaminated data
imputer = Imputation.MatrixCompletion.CDRec(ts_2.data)
imputer.impute()

# [OPTIONAL] save your results in a new Time Series object
ts_3 = TimeSeries().import_matrix(imputer.recov_data)

# 5. score the imputation with the raw_data
downstream_options = {"evaluator": "forecaster", "model": "prophet"}
imputer.score(ts_1.data, ts_3.data)  # upstream standard analysis
imputer.score(ts_1.data, ts_3.data, downstream=downstream_options)  # downstream advanced analysis

# 6. display the results
ts_3.print_results(imputer.metrics, algorithm=imputer.algorithm)
ts_3.print_results(imputer.downstream_metrics, algorithm=imputer.algorithm)