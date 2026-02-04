import pandas as pd
from sklearn.utils import resample

def oversample_dataframe(df, target_column='ClassId'):
    class_counts = df[target_column].value_counts()
    max_count = class_counts.max()

    dfs_balanced = []

    for class_id in df[target_column].unique():
        df_class = df[df[target_column] == class_id]
        
        df_upsampled = resample(
            df_class, 
            replace=True,
            n_samples=max_count, 
            random_state=123
        )
        dfs_balanced.append(df_upsampled)

    balanced_df = pd.concat(dfs_balanced)
    
    balanced_df = balanced_df.sample(frac=1, random_state=123).reset_index(drop=True)
    
    print(f"Rozmiar zbioru po oversamplingu: {len(balanced_df)}")
    return balanced_df