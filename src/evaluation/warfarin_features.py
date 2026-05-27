import numpy as np
import pandas as pd

WARFARIN_TARGET = 'Therapeutic Dose of Warfarin'

DEFAULT_LEAKAGE_PATTERNS = [
    'Subject Reached Stable Dose of Warfarin',
    'INR on Reported Therapeutic Dose of Warfarin',
]

DEFAULT_POST_TREATMENT_PATTERNS = [
    'Target INR',
    'Estimated Target INR Range Based on Indication',
]


def find_matching_columns(df, patterns, ignore_columns=None):
    ignore_columns = set(ignore_columns or [])
    matches = []
    for col in df.columns:
        if col in ignore_columns:
            continue
        col_lower = col.lower()
        if any(pattern.lower() in col_lower for pattern in patterns):
            matches.append(col)
    return matches


def find_leakage_columns(df, target_col=WARFARIN_TARGET, extra_patterns=None):
    patterns = list(DEFAULT_LEAKAGE_PATTERNS)
    if extra_patterns:
        patterns.extend(extra_patterns)
    return find_matching_columns(df, patterns, ignore_columns=[target_col])


def find_post_treatment_columns(df, target_col=WARFARIN_TARGET, extra_patterns=None):
    patterns = list(DEFAULT_POST_TREATMENT_PATTERNS)
    if extra_patterns:
        patterns.extend(extra_patterns)
    return find_matching_columns(df, patterns, ignore_columns=[target_col])


def infer_continuous_columns(df, target_col=WARFARIN_TARGET, min_unique=20):
    continuous_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        unique_count = series.nunique(dropna=True)
        values = set(series.dropna().unique())
        is_binary = values.issubset({0, 1, 0.0, 1.0, False, True})
        if unique_count >= min_unique and not is_binary:
            continuous_cols.append(col)
    return continuous_cols


def audit_warfarin_features(df, target_col=WARFARIN_TARGET):
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")

    leakage_cols = find_leakage_columns(df, target_col=target_col)
    post_treatment_cols = find_post_treatment_columns(df, target_col=target_col)
    continuous_cols = infer_continuous_columns(df, target_col=target_col)
    non_numeric_cols = [
        col for col in df.columns
        if col != target_col and not pd.api.types.is_numeric_dtype(df[col])
    ]

    return {
        'shape': df.shape,
        'target_col': target_col,
        'target_distribution': df[target_col].value_counts(dropna=False).sort_index(),
        'leakage_columns': leakage_cols,
        'post_treatment_columns': post_treatment_cols,
        'continuous_columns': continuous_cols,
        'non_numeric_columns': non_numeric_cols,
        'n_features_before_target_drop': df.shape[1] - 1,
    }


def print_feature_audit(audit):
    print('Data shape:', audit['shape'])
    print('Target column:', audit['target_col'])
    print('Number of features before target drop:', audit['n_features_before_target_drop'])
    print('\nTarget distribution:')
    print(audit['target_distribution'])
    print('\nLeakage columns to drop:')
    print(audit['leakage_columns'] if audit['leakage_columns'] else 'None found')
    print('\nPost-treatment columns found:')
    print(audit['post_treatment_columns'] if audit['post_treatment_columns'] else 'None found')
    print('\nContinuous columns to standardize:')
    print(audit['continuous_columns'] if audit['continuous_columns'] else 'None found')
    print('\nNon-numeric feature columns:')
    print(audit['non_numeric_columns'] if audit['non_numeric_columns'] else 'None found')


def prepare_warfarin_bandit_frame(
    df,
    target_col=WARFARIN_TARGET,
    drop_leakage=True,
    leakage_extra_patterns=None,
    standardize_continuous=True,
    continuous_cols=None,
    add_intercept=True,
    intercept_col='intercept',
):
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")

    clean_df = df.copy()
    info = audit_warfarin_features(clean_df, target_col=target_col)

    drop_cols = []
    if drop_leakage:
        drop_cols.extend(find_leakage_columns(clean_df, target_col, leakage_extra_patterns))
    drop_cols = list(dict.fromkeys(drop_cols))
    clean_df = clean_df.drop(columns=drop_cols, errors='ignore')

    if continuous_cols is None:
        continuous_cols = infer_continuous_columns(clean_df, target_col=target_col)
    else:
        continuous_cols = [col for col in continuous_cols if col in clean_df.columns and col != target_col]

    scaling_stats = {}
    if standardize_continuous and continuous_cols:
        for col in continuous_cols:
            mean = clean_df[col].mean()
            std = clean_df[col].std(ddof=0)
            if std == 0 or np.isnan(std):
                std = 1.0
            clean_df[col] = (clean_df[col] - mean) / std
            scaling_stats[col] = {'mean': mean, 'std': std}

    feature_cols = [col for col in clean_df.columns if col != target_col]
    clean_df[feature_cols] = clean_df[feature_cols].astype(float)
    clean_df[target_col] = clean_df[target_col].astype(int)
    clean_df = clean_df.copy()

    if add_intercept:
        intercept = pd.DataFrame({intercept_col: np.ones(len(clean_df), dtype=float)}, index=clean_df.index)
        clean_df = pd.concat([intercept, clean_df], axis=1)

    info.update({
        'dropped_columns': drop_cols,
        'continuous_columns_used': continuous_cols,
        'scaling_stats': scaling_stats,
        'added_intercept': add_intercept,
        'final_shape': clean_df.shape,
        'feature_columns': [col for col in clean_df.columns if col != target_col],
    })
    return clean_df, info


def split_features_target(df, target_col=WARFARIN_TARGET):
    X = df.drop(columns=[target_col]).to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=int)
    return X, y


def baseline_accuracy_from_columns(df, target_col, prediction_cols):
    rows = []
    for col in prediction_cols:
        if col not in df.columns:
            continue
        correct = df[col].astype(str) == df[target_col].astype(str)
        rows.append({
            'policy': col,
            'accuracy': correct.mean(),
            'fraction_incorrect': 1.0 - correct.mean(),
            'n_correct': int(correct.sum()),
            'n_total': int(len(correct)),
        })
    return pd.DataFrame(rows)
