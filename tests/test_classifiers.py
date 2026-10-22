import unittest

from recovery.downstream import Artifact


class TestClassifiers(unittest.TestCase):


    def test_classifier(self):
        """
        Verify if the downstream process is working
        """

        from imputegap.recovery.manager import TimeSeries
        from imputegap.recovery.downstream import Classifier
        from imputegap.tools import utils

        fixed_rate = 0.2

        # initialize the time series object
        ts = TimeSeries()
        print(f"\nImputeGAP datasets : {ts.datasets}")


        # load and normalize the dataset
        ts.load_classify_dataset("Beef")
        classify_sets = ts.load_proper_output_classify()

        # initialize the time series object
        ts = TimeSeries()
        print(f"\nImputeGAP datasets : {ts.datasets}")

        artifact = Artifact(dataset="Beef", model="knn", algorithm="cdrec", pattern="mcar")

        # load and normalize the dataset
        ts.load_classify_dataset(artifact.dataset, normalizer="z-score", to_numpy=True)
        classify_sets = ts.load_proper_output_classify()

        imputer, recovery, missing = utils.prepare_series_by_class(labels=classify_sets[1], raw_data=ts.data.copy(),
                                                                   imputer=artifact.algorithm, params=None,
                                                                   pattern=artifact.pattern,
                                                                   rate_dataset=artifact.fixed_x,
                                                                   rate_series=artifact.x, offset=artifact.offset,
                                                                   verbose=True)

        # compute and print the downstream results
        classifier = Classifier(task="classify",
                                model=artifact.model,
                                X_test=classify_sets[0],
                                y_train=classify_sets[1],
                                y_test=classify_sets[2],
                                algorithm=imputer.algorithm,
                                dataset=classify_sets[7],
                                to_cache=True,
                                use_cache=True,
                                artifact=artifact,
                                referential=True,
                                plots=False)

        imputer.score(ts.data, imputer.recov_data, downstream=classifier)
        ts.print_results(imputer.downstream_metrics, text="Downstream results")

        self.assertIsInstance(imputer.downstream_metrics, dict)
        self.assertTrue(len(imputer.downstream_metrics) > 0, "Metrics dict is empty")
        self.assertIn("f1_groundtruth", imputer.downstream_metrics)
        f1 = float(imputer.downstream_metrics["f1_groundtruth"])
        self.assertGreaterEqual(f1, 0.0, f"f1_groundtruth should be >= 0, got {f1}")
        self.assertLessEqual(f1, 100.0, f"f1_groundtruth should be <= 1, got {f1}")

    def test_classifier_arsenal(self):
        """
        Verify if the downstream process is working
        """

        from imputegap.recovery.manager import TimeSeries
        from imputegap.recovery.downstream import Classifier
        from imputegap.tools import utils

        fixed_rate = 0.2

        # initialize the time series object
        ts = TimeSeries()
        print(f"\nImputeGAP datasets : {ts.datasets}")


        # load and normalize the dataset
        ts.load_classify_dataset("Beef")
        classify_sets = ts.load_proper_output_classify()

        # initialize the time series object
        ts = TimeSeries()
        print(f"\nImputeGAP datasets : {ts.datasets}")

        artifact = Artifact(dataset="Beef", model="arsenal", algorithm="cdrec", pattern="mcar")

        # load and normalize the dataset
        ts.load_classify_dataset(artifact.dataset, normalizer="z-score", to_numpy=True)
        classify_sets = ts.load_proper_output_classify()

        imputer, recovery, missing = utils.prepare_series_by_class(labels=classify_sets[1], raw_data=ts.data.copy(),
                                                                   imputer=artifact.algorithm, params=None,
                                                                   pattern=artifact.pattern,
                                                                   rate_dataset=artifact.fixed_x,
                                                                   rate_series=artifact.x, offset=artifact.offset,
                                                                   verbose=True)

        # compute and print the downstream results
        classifier = Classifier(task="classify",
                                model=artifact.model,
                                X_test=classify_sets[0],
                                y_train=classify_sets[1],
                                y_test=classify_sets[2],
                                algorithm=imputer.algorithm,
                                dataset=classify_sets[7],
                                to_cache=False,
                                use_cache=False,
                                artifact=artifact,
                                referential=False,
                                plots=False)

        imputer.score(ts.data, imputer.recov_data, downstream=classifier)
        ts.print_results(imputer.downstream_metrics, text="Downstream results")

        self.assertIsInstance(imputer.downstream_metrics, dict)
        self.assertTrue(len(imputer.downstream_metrics) > 0, "Metrics dict is empty")
        self.assertIn("f1_imputer", imputer.downstream_metrics)
        f1 = float(imputer.downstream_metrics["f1_imputer"])
        self.assertGreaterEqual(f1, 0.0, f"f1_imputer should be >= 0, got {f1}")
        self.assertLessEqual(f1, 100.0, f"f1_imputer should be <= 1, got {f1}")





        