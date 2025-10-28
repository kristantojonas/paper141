import pandas as pd

# Column groups
ligcol = ['Cellulose (%)', 'Hemicellulose (%)', 'Lignin (%)']
ycol   = ['Gas', 'Liquid', 'Solid']
ucol   = ['Sulfur (%)', 'Carbon (%)', 'Hydrogen (%)', 'Nitrogen (%)', 'Oxygen (%)']
pcol   = ['Ash (%)', 'Volatiles (%)', 'Fixed Carbon (%)']
moisture_col = 'Moisture (%)'  # for fixed carbon group

# --- Normalization Functions ---

def normalize_fully_present(df, columns):
    df_copy = df.copy()
    for i in df.index:
        row = df_copy.loc[i, columns]
        if row.isna().any():
            continue
        total = row.sum()
        if total == 0:
            continue
        normalized = row / total * 100
        df_copy.loc[i, columns] = normalized.values
    return df_copy

def normalize_sulfur_optional(df, columns):
    df_copy = df.copy()
    for i in df.index:
        row = df_copy.loc[i, columns]
        if row.isna().sum() == 1 and pd.isna(row['Sulfur (%)']):
            values = row.drop('Sulfur (%)')
            total = values.sum()
            if total == 0:
                continue
            normalized = values / total * 100
            for col in values.index:
                df_copy.loc[i, col] = normalized[col]
        elif not row.isna().any():
            total = row.sum()
            if total == 0:
                continue
            normalized = row / total * 100
            df_copy.loc[i, columns] = normalized.values
    return df_copy

def normalize_except_moisture(df, target_cols, moisture_col='Moisture (%)'):
    df_copy = df.copy()
    for i in df.index:
        if pd.isna(df_copy.loc[i, moisture_col]):
            continue  # Moisture must be present

        row = df_copy.loc[i, target_cols]
        if row.isna().sum() > 1:
            continue  # Only allow at most one NaN

        non_nan = row.dropna()
        total = non_nan.sum()
        if total == 0:
            continue
        normalized = non_nan / total * 100
        for col in non_nan.index:
            df_copy.loc[i, col] = normalized[col]
    return df_copy

# --- Apply all normalizations ---

def PREPREPROCESSING(df):
    df = normalize_fully_present(df, ycol)
    df = normalize_fully_present(df, ligcol)
    df = normalize_sulfur_optional(df, ucol)
    df = normalize_except_moisture(df, pcol, moisture_col)
    return df