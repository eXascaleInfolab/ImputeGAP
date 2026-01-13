import time
from itertools import product
import numpy as np
from imputegap.recovery.imputation import Imputation
from imputegap.tools import utils
from imputegap.tools.algorithm_parameters import SEARCH_SPACES, ALL_ALGO_PARAMS, PARAM_NAMES, SEARCH_SPACES_PSO, RAYTUNE_PARAMS
import imputegap.tools.algorithm_parameters as sh_params


class BaseOptimizer:
    """
    A base class for optimization of imputation algorithm hyperparameters.

    Provides structure and common functionality for different optimization strategies.

    Methods
    -------
    _objective(**kwargs):
        Abstract method to evaluate the imputation algorithm with the provided parameters. Must be implemented by subclasses.

    optimize(input_data, incomp_data, metrics, algorithm, **kwargs):
        Abstract method for the main optimization process. Must be implemented by subclasses.
    """

    def __init__(self):
        pass

    def _objective(self, **kwargs):
        """
        Abstract objective function for optimization.

        This method evaluates the imputation algorithm with the provided parameters and computes the error
        across the selected metrics. The exact implementation depends on the optimization method.

        Since different optimization methods (e.g., Particle Swarm, Bayesian) may require different inputs,
        the parameters of this function are passed as keyword arguments (**kwargs). Subclasses should
        implement this method with the required parameters for the specific optimization.

        Parameters
        ----------
        **kwargs : dict
            Parameters needed to evaluate the imputation algorithm, such as:
            - input_data : numpy.ndarray
                The ground truth time series dataset.
            - contamination : numpy.ndarray
                The contaminated time series dataset to impute.
            - algorithm : str
                The imputation algorithm name.
            - metrics : list of str
                List of selected metrics for optimization.
            - params : dict or list
                Parameter values for the optimization.

        Returns
        -------
        float
            Mean error for the selected metrics.
        """
        raise NotImplementedError("Subclasses must implement the _objective method")

    def optimize(self, input_data, incomp_data, metrics, algorithm, **kwargs):
        """
        Abstract method for optimization. Must be implemented in subclasses.

        This method performs the optimization of hyperparameters for a given imputation algorithm. Each subclass
        implements a different optimization strategy (e.g., Greedy, Bayesian, Particle Swarm) and uses the
        `_objective` function to evaluate the parameters.

        Parameters
        ----------
        input_data : numpy.ndarray
            The ground truth time series dataset.
        incomp_data : numpy.ndarray
            The contaminated time series dataset to impute.
        metrics : list of str
            List of selected metrics for optimization.
        algorithm : str
            The imputation algorithm to optimize.
        **kwargs : dict
            Additional parameters specific to the optimization strategy (e.g., number of iterations, particles, etc.).

        Returns
        -------
        tuple
            A tuple containing the best parameters and their corresponding score.
        """
        raise NotImplementedError("Subclasses must implement the optimize method")


class Optimization:
    """
    A class for performing optimization of imputation algorithm hyperparameters.

    This class contains methods for various optimization strategies such as Greedy, Bayesian, Particle Swarm,
    and Successive Halving, used to find the best parameters for different imputation algorithms.

    Methods
    -------
    Greedy.optimize(input_data, incomp_data, metrics=["RMSE"], algorithm="cdrec", n_calls=250):
        Perform greedy optimization for hyperparameters.

    Bayesian.optimize(input_data, incomp_data, metrics=["RMSE"], algorithm="cdrec", n_calls=100, n_random_starts=50, acq_func='gp_hedge'):
        Perform Bayesian optimization for hyperparameters.

    ParticleSwarm.optimize(input_data, incomp_data, metrics, algorithm, n_particles, c1, c2, w, iterations, n_processes):
        Perform Particle Swarm Optimization (PSO) for hyperparameters.

    SuccessiveHalving.optimize(input_data, incomp_data, metrics, algorithm, num_configs, num_iterations, reduction_factor):
        Perform Successive Halving optimization for hyperparameters.
    """

    class Greedy(BaseOptimizer):
        """
        Greedy optimization strategy for hyperparameters.
        """

        def _objective(self, input_data, incomp_data, algorithm, metrics, params):
            """
            Objective function for Greedy optimization.

            Parameters
            ----------
            input_data : numpy.ndarray
                The ground truth time series dataset.
            incomp_data : numpy.ndarray
                The contaminated time series dataset to impute.
            algorithm : str
                The imputation algorithm name.
            metrics : list of str
                List of selected metrics for optimization.
            params : dict
                The parameters for the imputation algorithm.

            Returns
            -------
            float
                Mean error for the selected metrics.
            """
            errors = Imputation.evaluate_params(input_data, incomp_data, params, algorithm)

            if not isinstance(metrics, list):
                metrics = [metrics]

            return np.mean([errors[metric] for metric in metrics])

        def optimize(self, input_data, incomp_data, metrics=["RMSE"], algorithm="cdrec", n_calls=250):
            """
            Perform greedy optimization for hyperparameters.

            Parameters
            ----------
            input_data : numpy.ndarray
                The ground truth time series dataset.
            incomp_data : numpy.ndarray
                The contaminated time series dataset to impute.
            metrics : list of str, optional
                List of selected metrics for optimization (default is ["RMSE"]).
            algorithm : str, optional
                The imputation algorithm to optimize (default is 'cdrec').
            n_calls : int, optional
                Number of calls to the objective function (default is 250).

            Returns
            -------
            tuple
                A tuple containing the best parameters and their corresponding score.
            """
            start_time = time.time()  # Record start time

            # Map the parameter ranges to the algorithm-specific search space
            param_ranges = ALL_ALGO_PARAMS[algorithm]

            # Extract parameter names and their ranges for the selected algorithm
            param_names = list(param_ranges.keys())
            param_values = list(param_ranges.values())

            # Generate all combinations of parameters in the search space
            param_combinations = list(product(*param_values))  # Cartesian product of all parameter values

            # Placeholder for the best parameters and their score
            best_params = None
            best_score = float('inf')  # Assuming we are minimizing the objective function

            run_count = 0
            # Conduct greedy optimization over parameter combinations
            for params in param_combinations:

                if n_calls is not None and run_count >= n_calls:
                    break

                # Convert params to a dictionary for compatibility
                params_dict = {name: value for name, value in zip(param_names, params)}

                # Calculate the score for the current set of parameters
                score = self._objective(input_data, incomp_data, algorithm, metrics, params_dict)

                # Update the best parameters if the current score is better
                if score < best_score:
                    best_score = score
                    best_params = params_dict

                print(f"{score = }")

                # Increment the run counter
                run_count += 1

            end_time = time.time()
            print(f"\n> logs: optimization greedy - Execution Time: {(end_time - start_time):.4f} seconds\n")

            return best_params, best_score

    class Bayesian(BaseOptimizer):
        """
        Bayesian optimization strategy for hyperparameters.
        """

        def _objective(self, input_data, incomp_data, algorithm, metrics, params):
            """
            Objective function for Bayesian optimization.

            Parameters
            ----------
            input_data : numpy.ndarray
                The ground truth time series dataset.
            incomp_data : numpy.ndarray
                The contaminated time series dataset to impute.
            algorithm : str
                The imputation algorithm name.
            metrics : list of str
                List of selected metrics for optimization.
            params : dict
                Parameter values for the optimization.

            Returns
            -------
            float
                Mean error for the selected metrics.
            """
            # Check if params is a dictionary or a list
            if isinstance(params, dict):
                param_values = tuple(params.values())  # Convert dictionary to tuple of values
            else:
                param_values = tuple(params)

            if not isinstance(metrics, list):
                metrics = [metrics]

            errors = Imputation.evaluate_params(input_data, incomp_data, param_values, algorithm)
            return np.mean([errors[metric] for metric in metrics])

        def optimize(self, input_data, incomp_data, metrics=["RMSE"], algorithm="cdrec", n_calls=100,
                     n_random_starts=50, acq_func='gp_hedge'):
            """
            Perform Bayesian optimization for hyperparameters.

            Parameters
            ----------
            input_data : numpy.ndarray
                The ground truth time series dataset.
            incomp_data : numpy.ndarray
                The contaminated time series dataset to impute.
            metrics : list of str, optional
                List of selected metrics for optimization (default is ["RMSE"]).
            algorithm : str, optional
                The imputation algorithm to optimize (default is 'cdrec').
            n_calls : int, optional
                Number of calls to the objective function (default is 100).
            n_random_starts : int, optional
                Number of random initial points (default is 50).
            acq_func : str, optional
                Acquisition function for the Gaussian prior (default is 'gp_hedge').

            Returns
            -------
            tuple
                A tuple containing the best parameters and their corresponding score.
            """
            # BAYESIAN IMPORT
            import skopt
            from skopt.space import Integer

            start_time = time.time()  # Record start time

            search_spaces = SEARCH_SPACES

            # Adjust the search space for 'cdrec' based on incomp_data
            if algorithm == 'cdrec':
                max_rank = incomp_data.shape[1] - 1
                SEARCH_SPACES['cdrec'][0] = Integer(0, min(9, max_rank), name='rank')  # Update the rank range

            # Define the search space
            space = search_spaces[algorithm]

            # Conduct Bayesian optimization
            optimizer = skopt.Optimizer(dimensions=space, n_initial_points=n_random_starts, acq_func=acq_func)
            for i in range(n_calls):
                suggested_params = optimizer.ask()
                score = self._objective(input_data, incomp_data, algorithm, metrics, suggested_params)
                optimizer.tell(suggested_params, score)

            # Optimal parameters
            optimal_params = optimizer.Xi[np.argmin(optimizer.yi)]
            optimal_params_dict = {name: value for name, value in zip([dim.name for dim in space], optimal_params)}

            end_time = time.time()
            print(f"\n> logs: optimization bayesian - Execution Time: {(end_time - start_time):.4f} seconds\n")

            return optimal_params_dict, np.min(optimizer.yi)

    class ParticleSwarm(BaseOptimizer):
        """
        Particle Swarm Optimization (PSO) strategy for hyperparameters.
        """

        def _format_params(self, particle_params, algorithm):
            """
            Format parameters for the given algorithm.

            Parameters
            ----------
            particle_params : list
                List of particle parameters.
            algorithm : str
                The imputation algorithm name.

            Returns
            -------
            list
                Formatted list of parameters.
            """
            three_side = ["cdrec", "mrnn", "stmvl"]
            if algorithm in three_side:
                particle_params = [int(particle_params[0]), particle_params[1], int(particle_params[2])]
            if algorithm == 'iim':
                particle_params = [int(particle_params[0])]
            return particle_params

        def _objective(self, input_data, incomp_data, algorithm, metrics, params):
            """
            Objective function for Particle Swarm Optimization.

            Parameters
            ----------
            input_data : numpy.ndarray
                The ground truth time series dataset.
            incomp_data : numpy.ndarray
                The contaminated time series dataset to impute.
            algorithm : str
                The imputation algorithm name.
            metrics : list of str
                List of selected metrics for optimization.
            params : numpy.ndarray
                Parameter values for the optimization.

            Returns
            -------
            numpy.ndarray
                Array of error values for each particle.
            """

            n_particles = params.shape[0]  # Get the number of particles

            # Initialize array to hold the errors for each particle
            errors_for_all_particles = np.zeros(n_particles)

            for i in range(n_particles):  # Iterate over each particle
                particle_params = self._format_params(params[i], algorithm)  # Get the parameters for this particle
                errors = Imputation.evaluate_params(input_data, incomp_data, tuple(particle_params), algorithm)
                errors_for_all_particles[i] = np.mean([errors[metric] for metric in metrics])
            return errors_for_all_particles

        def optimize(self, input_data, incomp_data, metrics, algorithm, n_particles, c1, c2, w, iterations,
                     n_processes):
            """
            Perform Particle Swarm Optimization for hyperparameters.

            Parameters
            ----------
            input_data : numpy.ndarray
                The ground truth time series dataset.
            incomp_data : numpy.ndarray
                The contaminated time series dataset to impute.
            metrics : list of str, optional
                List of selected metrics for optimization (default is ["RMSE"]).
            algorithm : str, optional
                The imputation algorithm to optimize (default is 'cdrec').
            n_particles : int
                Number of particles used in PSO.
            c1 : float
                PSO parameter, personal learning coefficient.
            c2 : float
                PSO parameter, global learning coefficient.
            w : float
                PSO parameter, inertia weight.
            iterations : int
                Number of iterations for the optimization.
            n_processes : int
                Number of processes during optimization.

            Returns
            -------
            tuple
                A tuple containing the best parameters and their corresponding score.
            """
            from functools import partial
            import pyswarms as ps

            start_time = time.time()  # Record start time

            if not isinstance(metrics, list):
                metrics = [metrics]

            # Define the search space
            search_space = SEARCH_SPACES_PSO

            if algorithm == 'cdrec':
                max_rank = incomp_data.shape[1] - 1
                search_space['cdrec'][0] = (search_space['cdrec'][0][0], min(search_space['cdrec'][0][1], max_rank))

            # Select the correct search space based on the algorithm
            bounds = search_space[algorithm]

            # Convert search space to PSO-friendly format (two lists: one for min and one for max values for each parameter)
            lower_bounds, upper_bounds = zip(*bounds)
            bounds = (np.array(lower_bounds), np.array(upper_bounds))

            # Call instance of PSO
            optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=len(bounds[0]),
                                                options={'c1': c1, 'c2': c2, 'w': w}, bounds=bounds)

            # Perform optimization
            objective_with_args = partial(self._objective, input_data, incomp_data, algorithm, metrics)
            cost, pos = optimizer.optimize(objective_with_args, iters=iterations, n_processes=n_processes)

            param_names = PARAM_NAMES

            optimal_params = self._format_params(pos, algorithm)
            optimal_params_dict = {param_name: value for param_name, value in
                                   zip(param_names[algorithm], optimal_params)}

            end_time = time.time()
            print(f"\n> logs: optimization pso - Execution Time: {(end_time - start_time):.4f} seconds\n")

            return optimal_params_dict, cost

    class SuccessiveHalving(BaseOptimizer):

        def _objective(self, errors_dict, metrics):
            """
            Objective function for Successive Halving optimization.

            Parameters
            ----------
            errors_dict : dict
                Dictionary containing error metrics.
            metrics : list of str
                List of selected metrics for optimization.

            Returns
            -------
            float
                Mean error for the selected metrics.
            """
            selected_errors = [errors_dict[metric] for metric in metrics]
            return np.mean(selected_errors)

        def optimize(self, input_data, incomp_data, metrics, algorithm, num_configs, num_iterations,
                     reduction_factor):
            """
            Perform Successive Halving optimization for hyperparameters.

            Parameters
            ----------
            input_data : numpy.ndarray
                The ground truth time series dataset.
            incomp_data : numpy.ndarray
                The contaminated time series dataset to impute.
            metrics : list of str, optional
                List of selected metrics for optimization (default is ["RMSE"]).
            algorithm : str, optional
                The imputation algorithm to optimize (default is 'cdrec').
            num_configs : int
                Number of configurations to try.
            num_iterations : int
                Number of iterations for the optimization.
            reduction_factor : int
                Reduction factor for the number of configurations kept after each iteration.

            Returns
            -------
            tuple
                A tuple containing the best parameters and their corresponding score.
            """
            start_time = time.time()  # Record start time

            if not isinstance(metrics, list):
                metrics = [metrics]

            # Define the parameter names for each algorithm
            param_names = PARAM_NAMES

            data_length = len(input_data)
            chunk_size = data_length // num_iterations

            # prepare configurations for each algorithm separately
            if algorithm == 'cdrec':
                max_rank = incomp_data.shape[1] - 1
                temp_rank_range = [i for i in sh_params.CDREC_RANK_RANGE if i < max_rank]

                if not temp_rank_range:
                    raise ValueError("No suitable rank found within CDREC_RANK_RANGE for the given matrix shape!")

                configs = [(np.random.choice(temp_rank_range), np.random.choice(sh_params.CDREC_EPS_RANGE), np.random.choice(sh_params.CDREC_ITERS_RANGE)) for _ in range(num_configs)]
            elif algorithm == 'iim':
                configs = [(np.random.choice(sh_params.IIM_LEARNING_NEIGHBOR_RANGE))
                           for _ in range(num_configs)]
            elif algorithm == 'mrnn':
                configs = [(np.random.choice(sh_params.MRNN_HIDDEN_DIM_RANGE), np.random.choice(sh_params.MRNN_LEARNING_RATE_CHANGE), np.random.choice(sh_params.MRNN_NUM_ITER_RANGE)) for _ in range(num_configs)]
            elif algorithm == 'stmvl':
                configs = [(np.random.choice(sh_params.STMVL_WINDOW_SIZE_RANGE), np.random.choice(sh_params.STMVL_GAMMA_RANGE), np.random.choice(sh_params.STMVL_ALPHA_RANGE)) for _ in range(num_configs)]
            else:
                raise ValueError(f"Invalid algorithm: {algorithm}")

            for i in range(num_iterations):
                # Calculate how much data to use in this iteration
                end_idx = (i + 1) * chunk_size
                partial_input_data = input_data[:end_idx]
                partial_obfuscated = incomp_data[:end_idx]

                scores = [self._objective(
                    Imputation.evaluate_params(partial_input_data, partial_obfuscated, config, algorithm),
                    metrics) for config in configs]

                top_configs_idx = np.argsort(scores)[:max(1, len(configs) // reduction_factor)]
                configs = [configs[i] for i in top_configs_idx]
                if len(configs) <= 1:
                    break  # Exit the loop if only 1 configuration remains

            if not configs:
                raise ValueError("No configurations left after successive halving.")
            if algorithm == 'iim':
                best_config = min(configs, key=lambda single_config: self._objective(Imputation.evaluate_params(input_data, incomp_data, [single_config], algorithm), metrics))
            else:
                best_config = min(configs, key=lambda config: self._objective(
                    Imputation.evaluate_params(input_data, incomp_data, config, algorithm), metrics))

            best_score = self._objective(
                Imputation.evaluate_params(input_data, incomp_data, best_config, algorithm), metrics)

            # Check the size of param_names[algorithm]
            if len(param_names[algorithm]) == 1:
                # If only one parameter name, wrap best_config in a list if it's not already
                best_config = [best_config] if not isinstance(best_config, list) else best_config
            # Create the dictionary using zip
            best_config_dict = {name: value for name, value in zip(param_names[algorithm], best_config)}

            end_time = time.time()
            print(f"\n> logs: optimization sh - Execution Time: {(end_time - start_time):.4f} seconds\n")

            return best_config_dict, best_score

    class RayTune(BaseOptimizer):
        """
        RayTune optimization strategy for hyperparameters.
        """

        def _objective(self, params, input_data, incomp_data, algorithm, used_metric):
            """
            Objective function for RayTune optimization.
            """
            imputer = utils.config_impute_algorithm(incomp_data, algorithm, verbose=False)
            imputer.impute(user_def=True, params=params)

            print(f"{params = }")

            imputer.score(input_data=input_data)
            score = imputer.metrics.get(used_metric, "Key not found")
            return score

        def optimize(self, input_data, incomp_data, metrics=["RMSE"], algorithm="cdrec", n_calls=1, max_concurrent_trials=-1):
            """
            Perform Ray Tune optimization for hyperparameters.

            Parameters
            ----------
            input_data : numpy.ndarray
                The ground truth time series dataset.
            metrics : list of str, optional
                List of selected metrics for optimization (default is ["RMSE"]).
            algorithm : str, optional
                The imputation algorithm to optimize (default is 'cdrec').
            n_calls : int, optional
                Number of calls to the objective function (default is 10).
            max_concurrent_trials : int, optional
                Number of trials run in parallel, related to your total memory / cpu / gpu (default is 2).
                Please increase the value if you have more resources

            Returns
            -------
            tuple
                A tuple containing the best parameters and their corresponding score.
            """
            from ray import tune
            import ray

            if not ray.is_initialized():
                ray.init()
            used_metric = metrics[0]

            if max_concurrent_trials == -1:
                total_cpus = max(1, sum(node["Resources"].get("CPU", 0) for node in ray.nodes() if node["Alive"]) - 1)
                total_memory_gb = sum(node["Resources"].get("memory", 0) for node in ray.nodes() if node["Alive"]) / (1024 ** 3)

                print(f"\n\t\t(OPTI) > Ray Total accessible CPU cores for parallelization: {total_cpus}")
                print(f"\n\t\t(OPTI) > Ray Total accessible memory for parallelization: {total_memory_gb:.2f} GB")

                max_concurrent_trials = min(int(total_memory_gb // 2), total_cpus)

            print(f"\n\t\t(OPTI) > Ray tune max_concurrent_trials {max_concurrent_trials}, for {n_calls} calls and metric {used_metric}\n")

            start_time = time.time()  # Record start time

            search_space = RAYTUNE_PARAMS[algorithm]
            print(f"\n\t\t(OPTI) > Ray tune - SEARCH SPACE: {search_space}\n")

            def objective_wrapper(config):
                params = {key: config[key] for key in config}

                try:
                    score = self._objective(params, input_data, incomp_data, algorithm, used_metric)
                    if score is None or not isinstance(score, (int, float)):
                        raise ValueError("\n\n\n\t\t\tRAY_TUNE OBJECTIVE ERROR) >> Invalid score returned from _objective")
                except Exception as e:
                    print(f"\n\n\n\t\t\t(RAY_TUNE OBJECTIVE ERROR) >> Error in objective function: {e}")
                    score = float("inf")  # Return worst possible score

                return {used_metric: score}  # Ensures correct format

            analysis = tune.run(
                objective_wrapper,
                config=search_space,
                metric=used_metric,
                mode="min",
                num_samples=n_calls,
                max_concurrent_trials=max_concurrent_trials
            )

            print(f"\n(OPTI) > Ray tune - optimal parameters:\n\t{analysis.best_config}\n\n")

            end_time = time.time()
            print(f"\n> logs: optimization ray tune - Execution Time: {(end_time - start_time):.4f} seconds_____\n")

            ray.shutdown()

            return analysis.best_config
