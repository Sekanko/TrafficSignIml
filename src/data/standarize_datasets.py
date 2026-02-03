import pandas as pd
from PIL import Image
import os

def standardize_to_gtsrb(df):
    target_columns = ['Width', 'Height', 'Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2', 'ClassId', 'Path']
    
    df = df.copy()

    if df.empty:
        return pd.DataFrame(columns=target_columns)

    if 'Width' not in df.columns or 'Height' not in df.columns:
        widths = []
        heights = []
        
        for idx, row in df.iterrows():
            try:
                with Image.open(row['Path']) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except Exception:
                widths.append(0)
                heights.append(0)
        
        df['Width'] = widths
        df['Height'] = heights

    if 'Roi.X1' not in df.columns:
        df['Roi.X1'] = -1
    if 'Roi.Y1' not in df.columns:
        df['Roi.Y1'] = -1
    if 'Roi.X2' not in df.columns:
        df['Roi.X2'] = -1
    if 'Roi.Y2' not in df.columns:
        df['Roi.Y2'] = -1

    return df[target_columns]