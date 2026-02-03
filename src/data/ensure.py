import pandas as pd
from sklearn.utils import shuffle
from .german_dataset import get_german_dataframes
from .polish_dataset import get_polish_dataframes


def get_merged_data():
    g_train, g_val, g_test = get_german_dataframes()
    p_train, p_val, p_test = get_polish_dataframes()

    train_df = pd.concat([g_train, p_train], ignore_index=True)
    val_df = pd.concat([g_val, p_val], ignore_index=True)
    test_df = pd.concat([g_test, p_test], ignore_index=True)

    train_df = shuffle(train_df, random_state=50).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    target_columns = ['Width', 'Height', 'Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2', 'ClassId', 'Path']
    
    train_df = train_df[target_columns]
    val_df = val_df[target_columns]
    test_df = test_df[target_columns]

    return train_df, val_df, test_df