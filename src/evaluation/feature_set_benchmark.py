import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.evaluation.bandit_evaluation import evaluate_bandit, summarize_results


TARGET_COL = 'Therapeutic Dose of Warfarin'
ARM_LABELS = {0: 'low', 1: 'medium', 2: 'high'}
V2_OUTPUT_DIR = Path('output/preprocess_v2')
V2_MODELING_TABLE = 'warfarin_preprocess_v2_modeling_table.csv'
V2_FEATURE_SETS = 'warfarin_preprocess_v2_feature_sets.csv'


HEIGHT_WEIGHT_VARIANTS = {
    'knn_hw': 'knn_hw',
    'group_median_hw': 'group_median_hw',
    'regression_hw': 'regression_hw',
}


def resolve_repo_root(start=None):
    path = Path(start or Path.cwd()).resolve()
    for candidate in [path] + list(path.parents):
        if (candidate / 'src').exists() and (candidate / 'data').exists():
            return candidate
    raise FileNotFoundError('Could not find repository root containing src/ and data/.')


def load_v2_assets(repo_root=None, output_dir=V2_OUTPUT_DIR):
    repo_root = resolve_repo_root(repo_root)
    output_dir = repo_root / output_dir
    modeling_path = output_dir / V2_MODELING_TABLE
    feature_sets_path = output_dir / V2_FEATURE_SETS

    if not modeling_path.exists():
        raise FileNotFoundError(f'Missing V2 modeling table: {modeling_path}')
    if not feature_sets_path.exists():
        raise FileNotFoundError(f'Missing V2 feature-set manifest: {feature_sets_path}')

    df = pd.read_csv(modeling_path)
    feature_sets = pd.read_csv(feature_sets_path)
    return df, feature_sets, output_dir


def list_feature_sets(feature_sets):
    summary = (
        feature_sets.groupby('feature_set')['feature_name']
        .nunique()
        .reset_index(name='n_features')
        .sort_values(['n_features', 'feature_set'], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary


def get_feature_columns(feature_sets, feature_set_name):
    cols = feature_sets.loc[
        feature_sets['feature_set'] == feature_set_name, 'feature_name'
    ].tolist()
    if not cols:
        raise ValueError(f'Unknown or empty feature set: {feature_set_name}')
    return cols


def validate_feature_columns(df, feature_cols, feature_set_name='feature_set'):
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        preview = missing[:10]
        raise KeyError(
            f'{len(missing)} columns from {feature_set_name} are missing in the modeling table. '
            f'First missing columns: {preview}'
        )


def is_binary_series(series):
    values = set(pd.Series(series).dropna().unique())
    return values.issubset({0, 1, 0.0, 1.0, False, True})


def standardize_non_binary_features(X_df, skip_cols=None):
    X_df = X_df.copy()
    skip_cols = set(skip_cols or [])
    stats = []

    for col in X_df.columns:
        if col in skip_cols:
            continue
        if not pd.api.types.is_numeric_dtype(X_df[col]):
            continue
        if is_binary_series(X_df[col]):
            continue

        mean = X_df[col].mean()
        std = X_df[col].std(ddof=0)
        if pd.isna(std) or std == 0:
            std = 1.0
        X_df[col] = (X_df[col] - mean) / std
        stats.append({'feature': col, 'mean': float(mean), 'std': float(std)})

    return X_df, pd.DataFrame(stats)


def make_feature_matrix(
    df,
    feature_sets,
    feature_set_name,
    target_col=TARGET_COL,
    standardize=True,
    intercept_candidates=('Intercept', 'intercept'),
):
    feature_cols = get_feature_columns(feature_sets, feature_set_name)
    validate_feature_columns(df, feature_cols, feature_set_name)

    X_df = df[feature_cols].copy()
    y = df[target_col].astype(int).to_numpy()

    if standardize:
        X_df, scale_stats = standardize_non_binary_features(
            X_df,
            skip_cols=[col for col in intercept_candidates if col in X_df.columns],
        )
    else:
        scale_stats = pd.DataFrame(columns=['feature', 'mean', 'std'])

    X = X_df.to_numpy(dtype=float)
    return X, y, X_df.columns.tolist(), scale_stats


def detect_hw_variant(feature_set_name):
    if 'group_median' in feature_set_name:
        return 'group_median_hw'
    if 'regression' in feature_set_name:
        return 'regression_hw'
    return 'knn_hw'


def static_policy_predictions(df, feature_set_name):
    variant = detect_hw_variant(feature_set_name)
    clinical_col = f'Clinical Dose Bucket__{variant}'
    pgx_col = f'Pharmacogenetic Dose Bucket__{variant}'

    predictions = {'Fixed Medium Dose': np.ones(len(df), dtype=int)}
    if clinical_col in df.columns:
        predictions[f'Clinical Formula ({variant})'] = df[clinical_col].astype(int).to_numpy()
    if pgx_col in df.columns:
        predictions[f'Pharmacogenetic Formula ({variant})'] = df[pgx_col].astype(int).to_numpy()
    return predictions


def static_policy_summary(df, feature_set_names, target_col=TARGET_COL):
    y = df[target_col].astype(int).to_numpy()
    rows = []
    for feature_set_name in feature_set_names:
        for policy, pred in static_policy_predictions(df, feature_set_name).items():
            correct = pred == y
            rows.append({
                'feature_set': feature_set_name,
                'algorithm': policy,
                'accuracy': float(correct.mean()),
                'fraction_incorrect': float(1.0 - correct.mean()),
                'n_correct': int(correct.sum()),
                'n_total': int(len(y)),
            })
    return pd.DataFrame(rows)


def make_linucb_factory(linucb_cls, n_arms, n_features, alpha, lambda_reg):
    return lambda: linucb_cls(
        alpha=alpha,
        n_arms=n_arms,
        n_features=n_features,
        lambda_reg=lambda_reg,
    )


def evaluate_feature_set_grid(
    df,
    feature_sets,
    linucb_cls,
    feature_set_names,
    alphas,
    lambda_regs,
    seeds=range(5),
    standardize=True,
    progress=False,
    verbose=True,
):
    seeds = list(seeds)
    all_summaries = []
    all_results = []
    all_scale_stats = []

    for feature_set_name in feature_set_names:
        if verbose:
            print(f'Preparing feature set: {feature_set_name}')
        X, y, feature_cols, scale_stats = make_feature_matrix(
            df,
            feature_sets,
            feature_set_name,
            standardize=standardize,
        )
        n_arms = int(len(np.unique(y)))
        n_features = int(X.shape[1])
        scale_stats['feature_set'] = feature_set_name
        all_scale_stats.append(scale_stats)

        for alpha in alphas:
            for lambda_reg in lambda_regs:
                algorithm = f'RidgeLinUCB(alpha={alpha}, lambda={lambda_reg})'
                if verbose:
                    print(f'  Running {algorithm} with {len(seeds)} seeds...')
                run_results = evaluate_bandit(
                    name=algorithm,
                    bandit_factory=make_linucb_factory(
                        linucb_cls=linucb_cls,
                        n_arms=n_arms,
                        n_features=n_features,
                        alpha=alpha,
                        lambda_reg=lambda_reg,
                    ),
                    X=X,
                    y=y,
                    seeds=seeds,
                    progress=progress,
                )
                run_results['feature_set'] = feature_set_name
                run_results['alpha'] = alpha
                run_results['lambda_reg'] = lambda_reg
                run_results['n_features'] = n_features
                all_results.append(run_results)

                summary = summarize_results(run_results)
                summary['feature_set'] = feature_set_name
                summary['alpha'] = alpha
                summary['lambda_reg'] = lambda_reg
                summary['n_features'] = n_features
                all_summaries.append(summary)

    summary_df = pd.concat(all_summaries, ignore_index=True).sort_values(
        'final_accuracy_mean', ascending=False
    )
    results_df = pd.concat(all_results, ignore_index=True)
    scale_stats_df = pd.concat(all_scale_stats, ignore_index=True) if all_scale_stats else pd.DataFrame()
    return summary_df, results_df, scale_stats_df


def add_error_columns(results):
    out = results.copy()
    out['absolute_error'] = (out['chosen_arm'] - out['true_arm']).abs()
    out['severe_error'] = (out['absolute_error'] == 2).astype(int)
    out['one_step_error'] = (out['absolute_error'] == 1).astype(int)
    out['underdose_error'] = (out['chosen_arm'] < out['true_arm']).astype(int)
    out['overdose_error'] = (out['chosen_arm'] > out['true_arm']).astype(int)
    return out


def summarize_error_metrics(results):
    out = add_error_columns(results)
    keys = ['feature_set', 'algorithm', 'alpha', 'lambda_reg', 'run']
    keys = [col for col in keys if col in out.columns]
    by_run = out.groupby(keys).agg(
        accuracy=('reward', 'mean'),
        error_rate=('regret', 'mean'),
        severe_error_rate=('severe_error', 'mean'),
        one_step_error_rate=('one_step_error', 'mean'),
        underdose_rate=('underdose_error', 'mean'),
        overdose_rate=('overdose_error', 'mean'),
    ).reset_index()

    group_keys = [col for col in keys if col != 'run']
    summary = by_run.groupby(group_keys).agg(
        n_runs=('run', 'nunique'),
        accuracy_mean=('accuracy', 'mean'),
        accuracy_std=('accuracy', 'std'),
        severe_error_rate_mean=('severe_error_rate', 'mean'),
        severe_error_rate_std=('severe_error_rate', 'std'),
        one_step_error_rate_mean=('one_step_error_rate', 'mean'),
        underdose_rate_mean=('underdose_rate', 'mean'),
        overdose_rate_mean=('overdose_rate', 'mean'),
    ).reset_index()
    summary['accuracy_ci95'] = 1.96 * summary['accuracy_std'].fillna(0) / np.sqrt(summary['n_runs'])
    summary['severe_error_rate_ci95'] = 1.96 * summary['severe_error_rate_std'].fillna(0) / np.sqrt(summary['n_runs'])
    return summary.sort_values('accuracy_mean', ascending=False)


def classwise_metrics(results, labels=(0, 1, 2), label_names=ARM_LABELS):
    rows = []
    group_cols = ['feature_set', 'algorithm', 'alpha', 'lambda_reg', 'run']
    group_cols = [col for col in group_cols if col in results.columns]

    for key, group in results.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))
        y_true = group['true_arm'].astype(int).to_numpy()
        y_pred = group['chosen_arm'].astype(int).to_numpy()
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(labels),
            zero_division=0,
        )
        for i, label in enumerate(labels):
            rows.append({
                **base,
                'class_id': int(label),
                'class_name': label_names.get(int(label), str(label)),
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i]),
            })

    by_run = pd.DataFrame(rows)
    summary_keys = [col for col in group_cols if col != 'run'] + ['class_id', 'class_name']
    summary = by_run.groupby(summary_keys).agg(
        n_runs=('run', 'nunique'),
        precision_mean=('precision', 'mean'),
        precision_std=('precision', 'std'),
        recall_mean=('recall', 'mean'),
        recall_std=('recall', 'std'),
        f1_mean=('f1', 'mean'),
        f1_std=('f1', 'std'),
        support_mean=('support', 'mean'),
    ).reset_index()
    summary['recall_ci95'] = 1.96 * summary['recall_std'].fillna(0) / np.sqrt(summary['n_runs'])
    summary['f1_ci95'] = 1.96 * summary['f1_std'].fillna(0) / np.sqrt(summary['n_runs'])
    return summary.sort_values(summary_keys)


def confusion_table(results, labels=(0, 1, 2), label_names=ARM_LABELS, normalize='true'):
    rows = []
    group_cols = ['feature_set', 'algorithm', 'alpha', 'lambda_reg']
    group_cols = [col for col in group_cols if col in results.columns]

    for key, group in results.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))
        cm = confusion_matrix(
            group['true_arm'].astype(int),
            group['chosen_arm'].astype(int),
            labels=list(labels),
            normalize=normalize,
        )
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                rows.append({
                    **base,
                    'true_class_id': int(true_label),
                    'true_class_name': label_names.get(int(true_label), str(true_label)),
                    'predicted_class_id': int(pred_label),
                    'predicted_class_name': label_names.get(int(pred_label), str(pred_label)),
                    'value': float(cm[i, j]),
                    'normalization': normalize,
                })
    return pd.DataFrame(rows)


def select_top_feature_sets(summary, top_n=3):
    best = summary.sort_values('final_accuracy_mean', ascending=False)
    return best.drop_duplicates('feature_set').head(top_n)['feature_set'].tolist()


def save_benchmark_outputs(output_dir, **tables):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, table in tables.items():
        if table is None or len(table) == 0:
            continue
        path = output_dir / f'{name}.csv'
        table.to_csv(path, index=False)
        written[name] = str(path)
    with open(output_dir / 'benchmark_outputs.json', 'w') as f:
        json.dump(written, f, indent=2)
    return written


def plot_feature_set_accuracy(summary, top_n=None, title='Feature-set RidgeLinUCB comparison'):
    plot_df = summary.sort_values('final_accuracy_mean', ascending=False).copy()
    if top_n:
        plot_df = plot_df.head(top_n)
    labels = plot_df['feature_set'] + '\nα=' + plot_df['alpha'].astype(str) + ', λ=' + plot_df['lambda_reg'].astype(str)

    fig, ax = plt.subplots(figsize=(12, max(5, 0.5 * len(plot_df))))
    ax.barh(np.arange(len(plot_df)), plot_df['final_accuracy_mean'])
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Mean final accuracy')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.25)
    plt.tight_layout()
    return fig, ax
