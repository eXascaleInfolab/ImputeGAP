from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.explainer import Explainer
from imputegap.recovery.downstream import Artifact


# initialize the time series object
ts = TimeSeries()
exp = Explainer()
results=[]
print(f"\nImputeGAP downstream models for forcasting : {ts.forecasters}")
for ds in ["czelan", "electricity", "etth1", "etth2", "ili", "nn5", "nyse", "wike2000", "wind_speed", "airq", "atm", "beijing_traffic", "climate", "economics", "human_access", "paris"]:
    artifact = Artifact(dataset=ds, model="exp-smoothing", algorithm="cdrec", pattern="mcar", horizon=0)

    # load and normalize the dataset
    ts.load_forecasting_dataset(artifact.dataset, normalizer=None, verbose=True, horizon=artifact.horizon)
    #y_train, season = ts.load_proper_output_forecasting()

    r = exp.feature_correlation(data=ts.data)
    results.append(r)

    for result in r:
        print(f"{ds} ({ts.data.shape})= {result=}")

print("\n")
for (s, v) in results:
    print(v)