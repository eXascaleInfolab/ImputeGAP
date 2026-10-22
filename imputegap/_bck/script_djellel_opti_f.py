from recovery.optimization import Optimization

task_cla, datasets_cla = "c", ['Adiac', 'ArrowHead', 'Beef', 'BirdChicken', 'CBF', 'Car', 'CinCECGTorso', 'Colposcopy', 'Computers', 'CricketX', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'ECGFiveDays', 'EOGHorizontalSignal', 'EOGVerticalSignal', 'Earthquakes', 'EthanolLevel', 'FaceFour', 'FacesUCR', 'Fish', 'FreezerRegularTrain', 'FreezerSmallTrain', 'GunPoint', 'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'Ham', 'Haptics', 'Herring', 'InlineSkate', 'InsectEPGRegularTrain', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7', 'Meat', 'MedicalImages', 'MoteStrain', 'OSULeaf', 'OliveOil', 'PhalangesOutlinesCorrect', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxOutlineCorrect', 'RefrigerationDevices', 'Rock', 'ScreenType', 'SemgHandGenderCh2', 'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShapeletSim', 'SharePriceIncrease', 'SmallKitchenAppliances', 'SonyAIBORobotSurface1', 'Strawberry', 'SwedishLeaf', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'TwoPatterns', 'UMD', 'Wafer', 'Wine', 'Worms', 'WormsTwoClass', 'Yoga']
task_for, datasets_for = "f", ['airq', 'atm', 'beijing_traffic', 'climate', 'czelan', 'economics', 'electricity', 'etth1', 'etth2', 'human_access', 'ili', 'nn5', 'nyse', 'paris', 'wike2000', 'wind_speed']

task_cla, datasets_cla = "c", ['Beef']
task_for, datasets_for = "f", ['paris']

Optimization.optimize_adv_imputation_algo_upstream(task=task_cla, datasets=datasets_cla)
Optimization.optimize_adv_imputation_algo_upstream(task=task_for, datasets=datasets_for)