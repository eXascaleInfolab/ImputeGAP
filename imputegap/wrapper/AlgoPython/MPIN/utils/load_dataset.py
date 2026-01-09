import pickle
import os
import sys

import numpy as np
import pandas as pd

sys.path.append("..")

np.random.seed(26)

def get_model_size(model):
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    #print('Model Size: {:.3f} MB'.format(size_all_mb))
    return size_all_mb


def load_imputegap_dataset(data, window=2, seq_len=24, stream=1, shuffle=False, verbose=True):
    if verbose:
        print("\nloading imputegap dataset for MPIN...")
    return data

def load_imputegap_dataset_seq(data, window=2, seq_len=24, stream=1, shuffle=False):

    print("\nloading imputegap dataset for MPIN...\n")
    return data

    data = utils_imp.dataset_add_dimensionality(data, seq_len, reshapable=True, adding_nans=True, verbose=True, deep_verbose=False)

    num_samples = data.shape[0]
    num_features = data.shape[1]
    X = data
    print('sum of nan:', np.sum(np.isnan(X)))

    X = X.reshape(num_samples, seq_len, -1)

    print('X.reshape(num_samples, seq_len, -1)', X.shape, "\n\n")  # X shape (575424, 37)

    if shuffle:
        np.random.shuffle(X)

    X = X[:int(num_samples*stream),:window, :]

    print('X = X[:int(num_samples*stream),:window, :]', X.shape, "\n\n")  # X shape (575424, 37)

    X = X.reshape(-1, num_features)

    print('X.reshape(-1, num_features)', X.shape, "\n\n")  # X shape (575424, 37)

    return X


def load_saved_dataset(dataset_dir):
    """
    Load a dataset saved with:
        saving_into_h5(dataset_saving_dir, processed_data, classification_dataset=True)
        pickle_dump(scaler, os.path.join(dataset_saving_dir, 'scaler'))
    Returns:
        data (dict): containing numpy arrays / pandas DataFrames for train/val/test
        scaler: the fitted scaler object
    """
    import h5py

    h5_path = os.path.join(dataset_dir, "datasets.h5")
    scaler_path = os.path.join(dataset_dir, "scaler")

    # --- Load dataset from HDF5 ---
    with h5py.File(h5_path, "r") as f:
        print("Groups in dataset.h5:", list(f.keys()))
        data = {}
        for group_name, group in f.items():
            if isinstance(group, h5py.Group):
                data[group_name] = {k: v[()] for k, v in group.items()}
            elif isinstance(group, h5py.Dataset):
                data[group_name] = group[()]
        # optional: convert 'X' arrays to DataFrames if columns present
        if "columns" in f:
            cols = [c.decode() if isinstance(c, bytes) else str(c) for c in f["columns"][()]]
            if "X" in data:
                data["X"] = pd.DataFrame(data["X"], columns=cols)

    # --- Load scaler ---
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return data, scaler


def load_ICU_dataset(window=2, method='saits', stream=1, trans=True):

    print(f"testing: {window = }, {stream = }")

    X = run_dataset()
    num_samples = X.shape[0]

    if trans:
        X = X[:int(num_samples * stream), :window, :]

    return X.reshape(-1, 37)




def process_each_set(set_df, all_labels):
    # gene labels, y
    sample_ids = set_df["RecordID"].to_numpy().reshape(-1, 48)[:, 0]
    y = all_labels.loc[sample_ids].to_numpy().reshape(-1, 1)
    # gene feature vectors, X
    set_df = set_df.drop("RecordID", axis=1)
    feature_names = set_df.columns.tolist()
    X = set_df.to_numpy()
    X = X.reshape(len(sample_ids), 48, len(feature_names))
    return X, y, feature_names


def keep_only_features_to_normalize(all_feats, to_remove):
    for i in to_remove:
        all_feats.remove(i)
    return all_feats


def run_dataset(artificial_missing_rate=0.1, raw_data_path="RawData/Physio2012_mega/mega", outcome_files_dir="RawData/Physio2012_mega/", dataset_name="physio2012_37feats_01masked_1", saving_path="../generated_datasets"):
    from sklearn.preprocessing import StandardScaler

    here = os.path.dirname(os.path.abspath(__file__))

    saving_path = "generated_datasets"
    dataset_name = "imputegap"

    dataset_saving_dir = os.path.join(saving_path, dataset_name)
    # create saving dir
    if not os.path.exists(dataset_saving_dir):
        os.makedirs(dataset_saving_dir)


    outcome_files = ["Outcomes-a.txt", "Outcomes-b.txt", "Outcomes-c.txt"]
    outcome_collector = []
    for o_ in outcome_files:
        outcome_file_path = os.path.join(here, outcome_files_dir, o_)
        with open(outcome_file_path, "r") as f:
            outcome = pd.read_csv(f)[["In-hospital_death", "RecordID"]]
        outcome = outcome.set_index("RecordID")
        outcome_collector.append(outcome)
    all_outcomes = pd.concat(outcome_collector)

    all_recordID = []
    df_collector = []
    p = os.path.join(here, raw_data_path)
    for filename in os.listdir(p):
        recordID = int(filename.split(".txt")[0])
        with open(os.path.join(p, filename), "r") as f:
            df_temp = pd.read_csv(f)
        df_temp["Time"] = df_temp["Time"].apply(lambda x: int(x.split(":")[0]))
        df_temp = df_temp.pivot_table("Value", "Time", "Parameter")
        df_temp = df_temp.reset_index()  # take Time from index as a col
        if len(df_temp) == 1:
            continue
        all_recordID.append(recordID)  # only count valid recordID
        if df_temp.shape[0] != 48:
            missing = list(set(range(0, 48)).difference(set(df_temp["Time"])))
            missing_part = pd.DataFrame({"Time": missing})
            df_temp = pd.concat([df_temp, missing_part], ignore_index=False, sort=False)
            df_temp = df_temp.set_index("Time").sort_index().reset_index()
        df_temp = df_temp.iloc[
            :48
        ]  # only take 48 hours, some samples may have more records, like 49 hours
        df_temp["RecordID"] = recordID
        df_temp["Age"] = df_temp.loc[0, "Age"]
        df_temp["Height"] = df_temp.loc[0, "Height"]
        df_collector.append(df_temp)
    df = pd.concat(df_collector, sort=True)
    df = df.drop(["Age", "Gender", "ICUType", "Height"], axis=1)
    df = df.reset_index(drop=True)
    df = df.drop("Time", axis=1)  # dont need Time col

    all_features = df.columns.tolist()
    feat_no_need_to_norm = ["RecordID"]
    feats_to_normalize = keep_only_features_to_normalize(
        all_features, feat_no_need_to_norm
    )

    my_set = df[df["RecordID"].isin(all_recordID)]

    # standardization
    scaler = StandardScaler()
    my_set.loc[:, feats_to_normalize] = scaler.fit_transform(my_set.loc[:, feats_to_normalize])

    my_set_X, my_set_y, feature_names = process_each_set(my_set, all_outcomes)

    N, T, F = my_set_X.shape

    arr2d = my_set_X.reshape(N * T, F)

    export_path = os.path.join(dataset_saving_dir, "dataset.txt")
    np.savetxt(export_path, arr2d, fmt="%.6f", delimiter=" ")
    print(f"Dataset exported to {export_path} (shape={arr2d.shape})")

    return my_set_X


if __name__ == "__main__":

    X = load_ICU_dataset_l()
