#plit

from imputegap.recovery.imputation import Imputation
from imputegap.recovery.contamination import GenGap
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the time series object
ts = TimeSeries()
print(f"\nImputation algorithms : {ts.algorithms}")

djellel = ['CSDI', 'GPT4TS', 'TimesNet', 'PRISTI', 'NuwaTS']
alpha = ['DeepMVI', 'BayOTIDE', 'BitGraph', 'GAIN', 'CDRec', 'DynaMMo', 'STMVL', 'SoftImpute', 'XGBOOST', 'KNNImpute']
beta = ['SAITS', 'MissNet', 'GRIN', 'HKMFT', 'MICE', 'GROUSE', 'IIM', 'Interpolation', 'IterativeSVD', 'MeanImpute']
gamma = ['MPIN', 'Moment', 'MRNN', 'BRITS', 'MissForest', 'ROSL',  'SPIRIT', 'SVT',  'TKCM', 'TRMF']

# ["Beef", "BirdChicken", "Adiac", "Car", "Lightning2", "Haptics", "Worms", "InlineSkate", "OliveOil", "Rock", "Wafer"]


from imputegap.recovery.benchmark import Benchmark
from imputegap.tools import utils

# launch the evaluation
bench = Benchmark()
bench.eval_downstream_classification(classifiers=["arsenal", "cboss"],
                                     algorithms=['ZeroImpute', 'CDRec', 'MeanImpute', 'SoftImpute'],
                                     datasets= ['Beef', 'Car'],  # utils.get_directory_names(),
                                     patterns=["mcar"],
                                     x_axis=[0.2, 0.4, 0.8],
                                     metrics=None,
                                     normalizer="z-score",
                                     report_title="cleanimp_downstream_01",
                                     contamination_by_class=True,
                                     imputation_by_class=True,
                                     to_cache=True,
                                     use_cache=True,
                                     fixed_rate=0.2,
                                     run_downstream=True,
                                     evaluate_upstream=False,
                                     verbose=False)
