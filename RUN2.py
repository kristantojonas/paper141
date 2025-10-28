import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit

# ---------------------------
# 1. Normalization Groups & Functions
# ---------------------------
ligcol = ['Cellulose', 'Hemicellulose', 'Lignin']
ycol   = ['Gas', 'Liquid', 'Solid']
ucol   = ['Sulfur', 'Carbon', 'Hydrogen', 'Nitrogen', 'Oxygen']
pcol   = ['Ash', 'Volatiles', 'Fixed Carbon']
moisture_col = 'Moisture'

def normalize_fully_present(df, columns):
    df_copy = df.copy()
    for i in df.index:
        row = df_copy.loc[i, columns]
        if row.isna().any() or row.sum() == 0:
            continue
        df_copy.loc[i, columns] = (row / row.sum()) * 100
    return df_copy

def normalize_sulfur_optional(df, columns):
    df_copy = df.copy()
    for i in df.index:
        row = df_copy.loc[i, columns]
        if row.isna().sum() <= 1:
            values = row.dropna()
            total = values.sum()
            if total > 0:
                normalized = values / total * 100
                for col in values.index:
                    df_copy.loc[i, col] = normalized[col]
    return df_copy

def normalize_except_moisture(df, target_cols, moisture_col='Moisture'):
    df_copy = df.copy()
    for i in df.index:
        if pd.isna(df_copy.loc[i, moisture_col]):
            continue
        row = df_copy.loc[i, target_cols]
        if row.isna().sum() <= 1 and row.dropna().sum() > 0:
            non_nan = row.dropna()
            normalized = non_nan / non_nan.sum() * 100
            for col in non_nan.index:
                df_copy.loc[i, col] = normalized[col]
    return df_copy

def PREPREPROCESSING(df):
    df = normalize_fully_present(df, ycol)
    df = normalize_fully_present(df, ligcol)
    df = normalize_sulfur_optional(df, ucol)
    df = normalize_except_moisture(df, pcol, moisture_col)
    return df


# ---------------------------
# 2. Load & Preprocess Data
# ---------------------------
df = pd.read_excel('RAW ML.xlsx', skiprows=1)
df = df.drop(columns=['No.', 'First Author'], errors='ignore')
df = PREPREPROCESSING(df)


# ---------------------------
# 2.5 Hierarchical Taxonomic Imputation
# ---------------------------
def hierarchical_impute(df, cols, genus_col='Genus', family_col='Family'):
    df_imputed = df.copy()

    for col in cols:
        for i in df_imputed.index:
            if pd.isna(df_imputed.loc[i, col]):
                genus = df_imputed.loc[i, genus_col] if genus_col in df_imputed.columns else np.nan
                family = df_imputed.loc[i, family_col] if family_col in df_imputed.columns else np.nan

                # 1️⃣ Genus-level average
                if pd.notna(genus):
                    genus_mean = df_imputed.loc[df_imputed[genus_col] == genus, col].mean(skipna=True)
                    if not np.isnan(genus_mean):
                        df_imputed.loc[i, col] = genus_mean
                        continue

                # 2️⃣ Family-level average
                if pd.notna(family):
                    family_mean = df_imputed.loc[df_imputed[family_col] == family, col].mean(skipna=True)
                    if not np.isnan(family_mean):
                        df_imputed.loc[i, col] = family_mean
                        continue

                # 3️⃣ Global average
                global_mean = df_imputed[col].mean(skipna=True)
                if not np.isnan(global_mean):
                    df_imputed.loc[i, col] = global_mean

    return df_imputed


# Apply hierarchical imputation only to input (non-yield) features
impute_cols = ligcol + ucol + pcol + [moisture_col]
df = hierarchical_impute(df, impute_cols, genus_col='Genus', family_col='Family')


# ---------------------------
# 3. Normalize All Parameters (0–1 scale)
# ---------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


# ---------------------------
# 4. Linear & Power-Law Regression
# ---------------------------
def power_law(x, a, b):
    return a * np.power(x, b)

results = []
feature_cols = [c for c in numeric_cols if c not in ycol]

for y_var in ycol:
    for x_var in feature_cols:
        subset = df[[x_var, y_var]].dropna()
        if len(subset) < 8:
            continue

        X = subset[x_var].values
        y = subset[y_var].values

        # --- Linear Regression ---
        lin = LinearRegression()
        lin.fit(X.reshape(-1, 1), y)
        slope = lin.coef_[0]
        intercept = lin.intercept_
        r2_lin = lin.score(X.reshape(-1, 1), y)

        results.append({
            'Feature': x_var,
            'Yield': y_var,
            'Model': 'Linear',
            'R²': r2_lin,
            'Gradient/Slope': slope,
            'Intercept/a': intercept,
            'Exponent/b': np.nan
        })

        # --- Power-law Regression ---
        try:
            valid = (X > 0) & (y > 0)
            popt, _ = curve_fit(power_law, X[valid], y[valid], maxfev=10000)
            a, b = popt
            y_pred = power_law(X[valid], a, b)
            ss_res = np.sum((y[valid] - y_pred) ** 2)
            ss_tot = np.sum((y[valid] - np.mean(y[valid])) ** 2)
            r2_pow = 1 - (ss_res / ss_tot)
            results.append({
                'Feature': x_var,
                'Yield': y_var,
                'Model': 'Power-law',
                'R²': r2_pow,
                'Gradient/Slope': np.nan,
                'Intercept/a': a,
                'Exponent/b': b
            })
        except Exception:
            continue


# ---------------------------
# 5. Export Results
# ---------------------------
res_df = pd.DataFrame(results)
res_df = res_df.sort_values(by=['Yield', 'R²'], ascending=[True, False])

linear_df = res_df[res_df['Model'] == 'Linear']
powerlaw_df = res_df[res_df['Model'] == 'Power-law']

with pd.ExcelWriter('regression_results_with_coefficients2.xlsx', engine='openpyxl') as writer:
    linear_df.to_excel(writer, sheet_name='Linear Regression', index=False)
    powerlaw_df.to_excel(writer, sheet_name='Power-law Regression', index=False)
