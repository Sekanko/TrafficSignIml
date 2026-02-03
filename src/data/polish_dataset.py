import os
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .map_classes import get_polish_mapping
except ImportError:
    from map_classes import get_polish_mapping

def get_polish_dataframes(val_size=0.2, test_size=0.1, random_state=42):
    path = kagglehub.dataset_download("chriskjm/polish-traffic-signs-dataset")
    mapping = get_polish_mapping()
    data = []

    for root, _, files in os.walk(path):
        folder_name = os.path.basename(root)
        if folder_name in mapping:
            class_id = mapping[folder_name]
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    data.append({
                        'Path': os.path.join(root, file),
                        'Label': folder_name,
                        'ClassId': class_id
                    })

    full_df = pd.DataFrame(data)

    if full_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    train_val_df, test_df = train_test_split(
        full_df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=full_df['ClassId']
    )

    relative_val_size = val_size / (1 - test_size)
    
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=relative_val_size, 
        random_state=random_state, 
        stratify=train_val_df['ClassId']
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)