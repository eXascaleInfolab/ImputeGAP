import os
import toml


def display_title(title="Master Thesis", aut="Quentin Nater", lib="ImputeGAP", university="University Fribourg - exascale infolab"):
    print("=" * 100)
    print(f"{title} : {aut}")
    print("=" * 100)
    print(f"    {lib} - {university}")
    print("=" * 100)


def search_path(set_name="test"):
    """
    Find the accurate path for loading files of tests
    :return: correct file paths
    """

    filepath = "../imputegap/dataset/" + set_name + ".txt"

    if not os.path.exists(filepath):
        filepath = filepath[1:]

    return filepath


def get_save_path_asset():
    """
    Find the accurate path for saving files of tests
    :return: correct file paths
    """
    filepath = "../tests/assets"

    if not os.path.exists(filepath):
        filepath = filepath[1:]

    return filepath


def load_parameters(query: str = "default", algorithm: str = "cdrec", dataset: str = "chlorine", optimizer: str="b"):
    """
    Load default values of algorithms

    :param query : ('optimal' or 'default'), load default or optimal parameters for algorithms | default "default"
    :param algorithm : algorithm parameters to load | default "cdrec"

    :return: tuples of optimal parameters and the config of default values
    """

    filepath = ""
    if query == "default":
        filepath = "../env/default_values.toml"
    elif query == "optimal":
        filepath = "../params/optimal_parameters_"+str(optimizer)+"_"+str(dataset)+"_"+str(algorithm)+".toml"
    else:
        print("Query not found for this function ('optimal' or 'default')")

    if not os.path.exists(filepath):
        filepath = filepath[1:]

    with open(filepath, "r") as _:
        config = toml.load(filepath)

    if algorithm == "cdrec":
        truncation_rank = int(config['cdrec']['rank'])
        epsilon = config['cdrec']['epsilon']
        iterations = int(config['cdrec']['iteration'])
        return (truncation_rank, epsilon, iterations)
    elif algorithm == "stmvl":
        window_size = int(config['stmvl']['window_size'])
        gamma = float(config['stmvl']['gamma'])
        alpha = int(config['stmvl']['alpha'])
        return (window_size, gamma, alpha)
    elif algorithm == "iim":
        learning_neighbors = int(config['iim']['learning_neighbors'])
        algo_code = config['iim']['algorithm_code']
        return (learning_neighbors, algo_code)
    elif algorithm == "mrnn":
        hidden_dim = int(config['mrnn']['hidden_dim'])
        learning_rate = float(config['mrnn']['learning_rate'])
        iterations = int(config['mrnn']['iterations'])
        sequence_length = int(config['mrnn']['sequence_length'])
        return (hidden_dim, learning_rate, iterations, sequence_length)
    elif algorithm == "colors":
        colors = config['colors']['plot']
        return colors
    else :
        print("Default/Optimal config not found for this algorithm")
        return None


def verification_limitation(percentage, low_limit=0.01, high_limit=1.0):
    """
    Format the percentage given by the user.
    :param percentage: The percentage to be checked.
    :param low_limit: The lower limit of the acceptable percentage range.
    :param high_limit: The upper limit of the acceptable percentage range.
    :return: Adjusted percentage.
    """
    if low_limit <= percentage <= high_limit:
        return percentage  # No modification needed

    elif 1 <= percentage <= 100:
        print(f"The percentage {percentage} is between 1 and 100. Dividing by 100 to convert to a decimal.")
        return percentage / 100

    else:
        print("The percentage", percentage, "is out of the acceptable range", low_limit, "-", high_limit, ".")
        return percentage

def format_selection(ts, selection):
    """
    Format the selection of series based on keywords
    @author Quentin Nater

    :param selection: current selection of series
    :param ts: dataset to contaminate
    :return series_selected : correct format of selection series
    """
    if not selection:
        selection = ["*"]

    if selection == ["*"]:
        series_selected = []
        for i in range(0, ts.shape[0]):
            series_selected.append(str(i))
        return series_selected

    elif "-" in selection[0]:
        series_selected = []
        value = selection[0]
        ending = int(value[1:])
        for i in range(0, ts.shape[0] - ending):
            series_selected.append(str(i))
        return series_selected

    elif "+" in selection[0]:
        series_selected = []
        value = selection[0]
        starting = int(value[1:])
        for i in range(starting, ts.shape[0]):
            series_selected.append(str(i))
        return series_selected

    else:
        return selection