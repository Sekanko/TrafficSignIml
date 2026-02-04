import os
import pandas as pd
from PIL import Image
from sklearn.utils import shuffle
from .german_dataset import get_german_dataframes
from .polish_dataset import get_polish_dataframes
from .preprocess_img import crop_image_by_roi

def overwrite_images_with_crops(df):
    if df.empty:
        return df
    
    processed_count = 0
    skipped_count = 0

    for idx, row in df.iterrows():
        file_path = row['Path']
        
        x1, y1, x2, y2 = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
        
        if x1 == -1:
            skipped_count += 1
            continue

        try:
            img = Image.open(file_path).convert("RGB")
            
            new_img = crop_image_by_roi(img, x1, y1, x2, y2)
            
            if new_img is img:
                skipped_count += 1
                continue
            
            new_img.save(file_path)
            processed_count += 1

        except Exception as e:
            print(f"Błąd przy przetwarzaniu {file_path}: {e}")
            skipped_count += 1

    return df

def get_merged_data():
    g_train, g_val, g_test = get_german_dataframes()
    p_train, p_val, p_test = get_polish_dataframes()

    train_df = pd.concat([g_train, p_train], ignore_index=True)
    val_df = pd.concat([g_val, p_val], ignore_index=True)
    test_df = pd.concat([g_test, p_test], ignore_index=True)

    target_columns = ['Width', 'Height', 'Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2', 'ClassId', 'Path']
    
    train_df = train_df[target_columns]
    val_df = val_df[target_columns]
    test_df = test_df[target_columns]

    overwrite_images_with_crops(train_df)
    overwrite_images_with_crops(val_df)
    overwrite_images_with_crops(test_df)

    train_df = shuffle(train_df, random_state=50).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df