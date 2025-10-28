from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def prepare_data(
    df,
    y_col,
    y_cols_all,
    bs,
    max_trials=100,
    test_size=0.2,
    tolerance=0.01,
    exclude_hier_cols=None  # NEW argument
):
    # Strip column names to avoid issues with whitespace
    df.columns = df.columns.str.strip()

    # Ensure y_col is a list
    if isinstance(y_col, str):
        y_col = [y_col]

    # Drop rows with NaN in any of the target columns
    df = df.dropna(subset=y_col)

    # Extract y (DataFrame if multiple columns)
    y = df[y_col].copy()

    # Drop all y_cols_all from features (ignore if not present)
    X_df = df.drop(columns=y_cols_all, errors='ignore').copy()

    # Identify column types
    qualitative_cols = X_df.select_dtypes(include='object').columns
    quantitative_cols = X_df.select_dtypes(include=[np.number]).columns

    # Handle excluded columns
    if exclude_hier_cols is None:
        exclude_hier_cols = []
    hier_cols = [col for col in quantitative_cols if col not in exclude_hier_cols]

    # Fill NaNs hierarchically: Feedstock → Species → Genus → Family → Global mean
    if 'Feedstock' in X_df.columns and hier_cols:
        X_df[hier_cols] = X_df.groupby("Feedstock")[hier_cols].transform(
            lambda grp: grp.fillna(grp.mean())
        )
    if 'Species' in X_df.columns and hier_cols:
        X_df[hier_cols] = X_df.groupby("Species")[hier_cols].transform(
            lambda grp: grp.fillna(grp.mean())
        )
    if 'Genus' in X_df.columns and hier_cols:
        X_df[hier_cols] = X_df.groupby("Genus")[hier_cols].transform(
            lambda grp: grp.fillna(grp.mean())
        )
    if 'Family' in X_df.columns and hier_cols:
        X_df[hier_cols] = X_df.groupby("Family")[hier_cols].transform(
            lambda grp: grp.fillna(grp.mean())
        )

    # Final fallback: global mean (applies to ALL numeric columns, including excluded ones)
    X_df[quantitative_cols] = X_df[quantitative_cols].fillna(X_df[quantitative_cols].mean())

    # --- Drop Feedstock (after imputation) ---
    if 'Feedstock' in X_df.columns:
        X_df = X_df.drop(columns=['Feedstock'])
    if 'Species' in X_df.columns:
        X_df = X_df.drop(columns=['Species'])

    # --- Handle qualitative features ---
    # Convert yes/no to 1/0 first
    for col in qualitative_cols:
        if col in X_df.columns:
            if set(X_df[col].dropna().unique()).issubset({'yes', 'no', 'Yes', 'No'}):
                X_df[col] = X_df[col].str.lower().map({'yes': 1, 'no': 0})

    # For remaining categorical columns → one-hot encode
    remaining_qual_cols = X_df.select_dtypes(include='object').columns
    X_encoded = pd.get_dummies(X_df, columns=remaining_qual_cols, dtype=int)

    # Scale numeric features
    scaler = MinMaxScaler()
    X_encoded[quantitative_cols] = scaler.fit_transform(X_encoded[quantitative_cols])

    best_seed = bs
    # best_score = float('inf')

    # for seed in range(1, max_trials + 1):
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X_encoded, y, test_size=test_size, random_state=seed
    #     )
    #     # Calculate train/test stats and relative difference
    #     train_stats = y_train.describe().loc[['mean', 'std']]
    #     test_stats = y_test.describe().loc[['mean', 'std']]
    #     rel_diff = np.abs(train_stats - test_stats) / train_stats
    #     avg_rel_diff = rel_diff.mean().mean()

    #     if avg_rel_diff < best_score:
    #         best_score = avg_rel_diff
    #         best_seed = seed

    #     if best_score < tolerance:
    #         break

    # Final split with best seed
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=best_seed
    )

    # print(f"Best random seed: {best_seed} with average relative difference: {best_score:.4f}")
    return X_train, X_test, y_train, y_test

def prepare_data_wgf(
    df,
    y_col,
    y_cols_all,
    bs,
    max_trials=100,
    test_size=0.2,
    tolerance=0.01,
    exclude_hier_cols=None  # NEW argument
):
    # Strip column names to avoid issues with whitespace
    df.columns = df.columns.str.strip()

    # Ensure y_col is a list
    if isinstance(y_col, str):
        y_col = [y_col]

    # Drop rows with NaN in any of the target columns
    df = df.dropna(subset=y_col)

    # Extract y (DataFrame if multiple columns)
    y = df[y_col].copy()

    # Drop all y_cols_all from features (ignore if not present)
    X_df = df.drop(columns=y_cols_all, errors='ignore').copy()

    # Identify column types
    qualitative_cols = X_df.select_dtypes(include='object').columns
    quantitative_cols = X_df.select_dtypes(include=[np.number]).columns

    # Handle excluded columns
    if exclude_hier_cols is None:
        exclude_hier_cols = []
    hier_cols = [col for col in quantitative_cols if col not in exclude_hier_cols]

    # Fill NaNs hierarchically: Feedstock → Species → Genus → Family → Global mean
    if 'Feedstock' in X_df.columns and hier_cols:
        X_df[hier_cols] = X_df.groupby("Feedstock")[hier_cols].transform(
            lambda grp: grp.fillna(grp.mean())
        )
    if 'Species' in X_df.columns and hier_cols:
        X_df[hier_cols] = X_df.groupby("Species")[hier_cols].transform(
            lambda grp: grp.fillna(grp.mean())
        )

    # Final fallback: global mean (applies to ALL numeric columns, including excluded ones)
    X_df[quantitative_cols] = X_df[quantitative_cols].fillna(X_df[quantitative_cols].mean())

    # --- Drop Feedstock (after imputation) ---
    if 'Feedstock' in X_df.columns:
        X_df = X_df.drop(columns=['Feedstock'])
    if 'Species' in X_df.columns:
        X_df = X_df.drop(columns=['Species'])

    # --- Handle qualitative features ---
    # Convert yes/no to 1/0 first
    for col in qualitative_cols:
        if col in X_df.columns:
            if set(X_df[col].dropna().unique()).issubset({'yes', 'no', 'Yes', 'No'}):
                X_df[col] = X_df[col].str.lower().map({'yes': 1, 'no': 0})

    # For remaining categorical columns → one-hot encode
    remaining_qual_cols = X_df.select_dtypes(include='object').columns
    X_encoded = pd.get_dummies(X_df, columns=remaining_qual_cols, dtype=int)

    # Scale numeric features
    scaler = MinMaxScaler()
    X_encoded[quantitative_cols] = scaler.fit_transform(X_encoded[quantitative_cols])

    best_seed = bs
    # best_score = float('inf')

    # for seed in range(1, max_trials + 1):
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X_encoded, y, test_size=test_size, random_state=seed
    #     )
    #     # Calculate train/test stats and relative difference
    #     train_stats = y_train.describe().loc[['mean', 'std']]
    #     test_stats = y_test.describe().loc[['mean', 'std']]
    #     rel_diff = np.abs(train_stats - test_stats) / train_stats
    #     avg_rel_diff = rel_diff.mean().mean()

    #     if avg_rel_diff < best_score:
    #         best_score = avg_rel_diff
    #         best_seed = seed

    #     if best_score < tolerance:
    #         break

    # Final split with best seed
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=best_seed
    )

    # print(f"Best random seed: {best_seed} with average relative difference: {best_score:.4f}")
    return X_train, X_test, y_train, y_test