import argparse
from recovery.benchmark import Benchmark
from tools import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Run downstream forecasting benchmark")

    parser.add_argument(
        "--task",
        type=str,
        default="imputation",
        help="Call imputation, forecasting or classification"
    )

    parser.add_argument(
        "--caching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable cache"
    )

    parser.add_argument(
        "--downstream_mod",
        nargs="+",
        default=["chronos"],
        help="Forecasters to evaluate"
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Forecasting horizon"
    )

    parser.add_argument(
        "--imp_algs",
        nargs="+",
        default=["SAITS"],
        help="Imputation algorithms to evaluate"
    )

    parser.add_argument(
        "--datasets_type",
        type=str,
        default="forecasting",
        help="Call forecasting or classification"
    )

    parser.add_argument(
        "--datasets_list",
        nargs="+",
        default=["paris"],
        help="Datasets to evaluate"
    )

    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["mcar"],
        help="Patterns to evaluate"
    )

    parser.add_argument(
        "--miss_rate",
        nargs="+",
        default=[0.2],
        help="Contamination rates"
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        type=str,
        default=["RMSE", "DOWNSTREAM_SMAPE", "MI", "CORRELATION", "RUNTIME"],
        help="Enable or disable metrics"
    )

    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable plots"
    )

    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable verbose"
    )


    return parser.parse_args()


if __name__ == "__main__":

    print(f"\n> CleanImp Benchmark: Loading benchmark configuration and requirements...\n")

    args = parse_args()
    if args.miss_rate != ["all"]:
        args.miss_rate = [float(x) for x in args.miss_rate]

    d_cl = utils.get_datasets_classifiers(verbose=False)
    d_fo = utils.get_dataset_forecasters(directory='datasets/forecast/', verbose=False)
    valid_datasets = set(d_cl) | set(d_fo)| {"all"}
    invalid = [dataset for dataset in args.datasets_list if dataset not in valid_datasets]
    if invalid:
        raise ValueError(f"Unknown dataset(s): {invalid}")

    if args.task == "imputation": #=UPSTREAM============================================================================

        # Check if all args.datasets belong to classification or forecasting datasets
        #if all(dataset in d_cl for dataset in args.datasets):
        #    final_task = "classification"
        #elif all(dataset in d_fo for dataset in args.datasets_list):
        #    final_task = "forecasting"
        #else:
        #    raise ValueError("Datasets cannot mix classification and forecasting datasets.")

        if args.datasets_type == "forecasting":  #=UPSTREAM-FORECATING==========================================================
            # launch the evaluation
            bench = Benchmark()
            bench.eval_downstream_forecasting(forecasters=[],
                                              horizon=args.horizon,
                                              algorithms=args.imp_algs,  # utils.list_of_top_cleanimp_for_algorithms(),
                                              datasets=args.datasets_list,
                                              # "utils.get_dataset_forecasters(directory='datasets/forecast/'),
                                              patterns=args.patterns,
                                              x_axis=args.miss_rate,
                                              metrics=args.metrics,
                                              normalizer="z-score",
                                              report_title="cleanimp_benchmark_for_up",
                                              bypass_error=False,
                                              to_cache=args.caching,
                                              use_cache=args.caching,
                                              run_downstream=False,
                                              evaluate_upstream=True,
                                              fixed_rate=0.2,
                                              verbose=args.verbose,
                                              inner_plots=False,
                                              plots=False,
                                              generate_plot=args.plots,
                                              referential=False,
                                              nbr_series=10000,
                                              nbr_vals=10000)

        else:  #=UPSTREAM-CLASSIFICATION================================================================================
            # launch the evaluation
            bench = Benchmark()
            bench.eval_downstream_classification(classifiers=[],  # for all: utils.list_of_classifiers(),
                                                 algorithms=args.imp_algs,  # for all: utils.list_of_algorithms()
                                                 datasets=args.datasets_list,  # for all: utils.get_datasets_classifiers(),
                                                 patterns=args.patterns,
                                                 x_axis=args.miss_rate,
                                                 metrics=args.metrics,
                                                 normalizer="z-score",
                                                 report_title="cleanimp_benchmark_cl_up",
                                                 contamination_by_class=True,
                                                 imputation_by_class=True,
                                                 bypass_error=False,
                                                 evaluate_upstream=True,
                                                 to_cache=args.caching,
                                                 use_cache=args.caching,
                                                 run_downstream=False,
                                                 fixed_rate=0.2,
                                                 inner_plots=False,
                                                 referential=False,
                                                 plots=False,
                                                 generate_plot=args.plots,
                                                 verbose=args.verbose)


    elif args.task == "forecasting":  #=DOWNSTREAM-FORECASTING==========================================================
        # launch the evaluation
        bench = Benchmark()
        bench.eval_downstream_forecasting(forecasters=args.downstream_mod,
                                          horizon=args.horizon,
                                          algorithms=args.imp_algs,  # utils.list_of_top_cleanimp_for_algorithms(),
                                          datasets=args.datasets_list,
                                          # "utils.get_dataset_forecasters(directory='datasets/forecast/'),
                                          patterns=args.patterns,
                                          x_axis=args.miss_rate,
                                          metrics=args.metrics,
                                          normalizer="z-score",
                                          report_title="cleanimp_benchmark_for_down",
                                          bypass_error=False,
                                          to_cache=args.caching,
                                          use_cache=args.caching,
                                          run_downstream=True,
                                          evaluate_upstream=False,
                                          fixed_rate=0.2,
                                          verbose=args.verbose,
                                          inner_plots=False,
                                          plots=False,
                                          generate_plot=args.plots,
                                          referential=False,
                                          nbr_series=10000,
                                          nbr_vals=10000)

    else:  #=DOWNSTREAM-CLASSIFICATION==================================================================================
        bench = Benchmark()
        bench.eval_downstream_classification(classifiers=args.downstream_mod,  # for all: utils.list_of_classifiers(),
                                             algorithms=args.imp_algs,  # for all: utils.list_of_algorithms()
                                             datasets=args.datasets_list,  # for all: utils.get_datasets_classifiers(),
                                             patterns=args.patterns,
                                             x_axis=args.miss_rate,
                                             metrics=args.metrics,
                                             normalizer="z-score",
                                             report_title="cleanimp_benchmark_cl_down",
                                             contamination_by_class=True,
                                             imputation_by_class=True,
                                             bypass_error=False,
                                             evaluate_upstream=False,
                                             to_cache=args.caching,
                                             use_cache=args.caching,
                                             run_downstream=True,
                                             referential=False,
                                             fixed_rate=0.2,
                                             plots=False,
                                             generate_plot=args.plots,
                                             verbose=args.verbose)