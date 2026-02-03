import os

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split


def get_german_dataframes():
    path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")

    train_csv = os.path.join(path, "Train.csv")
    test_csv = os.path.join(path, "Test.csv")

    full_train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    full_train_df["Path"] = full_train_df["Path"].apply(lambda x: os.path.join(path, x))
    test_df["Path"] = test_df["Path"].apply(lambda x: os.path.join(path, x))

    train_df, val_df = train_test_split(
        full_train_df, test_size=0.2, random_state=50, stratify=full_train_df["ClassId"]
    )

    return train_df, val_df, test_df
