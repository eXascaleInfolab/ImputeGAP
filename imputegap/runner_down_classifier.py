from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.downstream import Classifier
from imputegap.recovery.downstream import Artifact
from imputegap.tools import utils


# initialize the time series object
ts = TimeSeries()
print(f"\nImputeGAP datasets : {ts.datasets}")

artifact = Artifact(dataset="Strawberry", model="arsenal", algorithm="trmf", pattern="mcar", x=0.4)

# load and normalize the dataset
ts.load_classify_dataset(artifact.dataset, normalizer="z-score", to_numpy=True)
classify_sets = ts.load_proper_output_classify()

imputer, recovery, missing = utils.prepare_series_by_class(labels=classify_sets[1],
                                                           raw_data=ts.data.copy(),
                                                           imputer=artifact.algorithm,
                                                           pattern=artifact.pattern,
                                                           rate_dataset=artifact.fixed_x,
                                                           rate_series=artifact.x,
                                                           offset=artifact.offset,
                                                           verbose=False)

# compute and print the downstream results
classifier = Classifier(task="classify",
                        model=artifact.model,
                        X_test=classify_sets[0],
                        y_train=classify_sets[1],
                        y_test=classify_sets[2],
                        algorithm=imputer.algorithm,
                        dataset=classify_sets[7],
                        to_cache=True,
                        use_cache=False,
                        artifact=artifact,
                        referential=True,
                        plots=True)

imputer.score(ts.data, imputer.recov_data, downstream=classifier)
ts.print_results(imputer.downstream_metrics, text="Downstream results")