import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.evaluation.bandit_evaluation import compute_reward, evaluate_bandit, summarize_results


TARGET_COL = 'Therapeutic Dose of Warfarin'
ARM_LABELS = {0: 'low', 1: 'medium', 2: 'high'}
V2_OUTPUT_DIR = Path('output/preprocess_v2')
V2_MODELING_TABLE = 'warfarin_preprocess_v2_modeling_table.csv'
V2_FEATURE_SETS = 'warfarin_preprocess_v2_feature_sets.csv'

FINAL_FEATURE_SETS = [
    'v2_strict_regression_hw_full_dummies',
    'v2_strict_knn_hw_full_dummies',
]

FINAL_RIDGE_GRID = {
    'alphas': [0.0, 0.01, 0.025, 0.05, 0.1],
    'lambda_regs': [5.0, 10.0, 20.0],
}

FINAL_HYBRID_GRID = [
    {'alpha': 0.0, 'lambda_arm': 10.0, 'lambda_shared': 10.0, 'alpha_shared': 0.0, 'shared_weight': 0.5},
    {'alpha': 0.0, 'lambda_arm': 10.0, 'lambda_shared': 20.0, 'alpha_shared': 0.0, 'shared_weight': 0.5},
    {'alpha': 0.01, 'lambda_arm': 10.0, 'lambda_shared': 10.0, 'alpha_shared': 0.01, 'shared_weight': 0.5},
    {'alpha': 0.01, 'lambda_arm': 10.0, 'lambda_shared': 20.0, 'alpha_shared': 0.01, 'shared_weight': 0.5},
    {'alpha': 0.025, 'lambda_arm': 10.0, 'lambda_shared': 10.0, 'alpha_shared': 0.025, 'shared_weight': 0.5},
    {'alpha': 0.01, 'lambda_arm': 20.0, 'lambda_shared': 10.0, 'alpha_shared': 0.01, 'shared_weight': 0.5},
    {'alpha': 0.01, 'lambda_arm': 10.0, 'lambda_shared': 10.0, 'alpha_shared': 0.01, 'shared_weight': 1.0},
    {'alpha': 0.025, 'lambda_arm': 10.0, 'lambda_shared': 20.0, 'alpha_shared': 0.025, 'shared_weight': 1.0},
]

REWARD_SCHEMES_TO_COMPARE = ['binary', 'ordinal']


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


def available_final_feature_sets(feature_sets, requested=None):
    requested = list(requested or FINAL_FEATURE_SETS)
    available = set(feature_sets['feature_set'].unique())
    return [name for name in requested if name in available]


def list_feature_sets(feature_sets):
    order = {name: i for i, name in enumerate(FINAL_FEATURE_SETS)}
    summary = (
        feature_sets.groupby('feature_set')['feature_name']
        .nunique()
        .reset_index(name='n_features')
    )
    summary['display_order'] = summary['feature_set'].map(order).fillna(999).astype(int)
    return summary.sort_values(['display_order', 'feature_set']).drop(columns='display_order').reset_index(drop=True)


def get_feature_columns(feature_sets, feature_set_name):
    cols = feature_sets.loc[feature_sets['feature_set'] == feature_set_name, 'feature_name'].tolist()
    if not cols:
        raise ValueError(f'Unknown or empty feature set: {feature_set_name}')
    return cols


def validate_feature_columns(df, feature_cols, feature_set_name='feature_set'):
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise KeyError(
            f'{len(missing)} columns from {feature_set_name} are missing in the modeling table. '
            f'First missing columns: {missing[:10]}'
        )


def is_binary_series(series):
    values = set(pd.Series(series).dropna().unique())
    return values.issubset({0, 1, 0.0, 1.0, False, True})


def standardize_non_binary_features(X_df, skip_cols=None):
    X_df = X_df.copy()
    skip_cols = set(skip_cols or [])
    stats = []

    for col in X_df.columns:
        if col in skip_cols or not pd.api.types.is_numeric_dtype(X_df[col]) or is_binary_series(X_df[col]):
            continue
        mean = X_df[col].mean()
        std = X_df[col].std(ddof=0)
        std = 1.0 if pd.isna(std) or std == 0 else std
        X_df[col] = (X_df[col] - mean) / std
        stats.append({'feature': col, 'mean': float(mean), 'std': float(std)})

    return X_df, pd.DataFrame(stats)


def make_feature_matrix(df, feature_sets, feature_set_name, target_col=TARGET_COL, standardize=True):
    feature_cols = get_feature_columns(feature_sets, feature_set_name)
    validate_feature_columns(df, feature_cols, feature_set_name)

    X_df = df[feature_cols].copy()
    y = df[target_col].astype(int).to_numpy()

    if standardize:
        X_df, scale_stats = standardize_non_binary_features(X_df, skip_cols=['Intercept'])
    else:
        scale_stats = pd.DataFrame(columns=['feature', 'mean', 'std'])

    return X_df.to_numpy(dtype=float), y, X_df.columns.tolist(), scale_stats


def detect_hw_variant(feature_set_name):
    return 'knn_hw' if 'knn' in feature_set_name else 'regression_hw'


def static_policy_predictions(df, feature_set_name):
    return {'Fixed Medium Dose': np.ones(len(df), dtype=int)}


def static_policy_summary(df, feature_set_names, target_col=TARGET_COL):
    y = df[target_col].astype(int).to_numpy()
    rows = []
    for feature_set_name in feature_set_names:
        for policy, pred in static_policy_predictions(df, feature_set_name).items():
            correct = pred == y
            ordinal_rewards = [compute_reward(p, t, reward_scheme='ordinal') for p, t in zip(pred, y)]
            rows.append({
                'feature_set': feature_set_name,
                'algorithm': policy,
                'accuracy': float(correct.mean()),
                'fraction_incorrect': float(1.0 - correct.mean()),
                'mean_ordinal_reward': float(np.mean(ordinal_rewards)),
                'n_correct': int(correct.sum()),
                'n_total': int(len(y)),
            })
    return pd.DataFrame(rows)


def make_linucb_factory(linucb_cls, n_arms, n_features, alpha, lambda_reg):
    return lambda: linucb_cls(alpha=alpha, n_arms=n_arms, n_features=n_features, lambda_reg=lambda_reg)


def evaluate_feature_set_grid(
    df,
    feature_sets,
    linucb_cls,
    feature_set_names=None,
    alphas=None,
    lambda_regs=None,
    seeds=range(5),
    reward_scheme='binary',
    standardize=True,
    progress=False,
    verbose=True,
):
    feature_set_names = available_final_feature_sets(feature_sets, feature_set_names)
    alphas = list(alphas or FINAL_RIDGE_GRID['alphas'])
    lambda_regs = list(lambda_regs or FINAL_RIDGE_GRID['lambda_regs'])
    seeds = list(seeds)
    all_summaries, all_results, all_scale_stats = [], [], []

    for feature_set_name in feature_set_names:
        X, y, feature_cols, scale_stats = make_feature_matrix(
            df, feature_sets, feature_set_name, standardize=standardize
        )
        n_arms, n_features = int(len(np.unique(y))), int(X.shape[1])
        scale_stats['feature_set'] = feature_set_name
        scale_stats['reward_scheme'] = str(reward_scheme)
        all_scale_stats.append(scale_stats)

        if verbose:
            print(f'RidgeLinUCB feature set: {feature_set_name} ({n_features} features), reward={reward_scheme}')

        for alpha in alphas:
            for lambda_reg in lambda_regs:
                algorithm = f'RidgeLinUCB(alpha={alpha}, lambda={lambda_reg})'
                if verbose:
                    print(f'  {algorithm}')
                result = evaluate_bandit(
                    name=algorithm,
                    bandit_factory=make_linucb_factory(linucb_cls, n_arms, n_features, alpha, lambda_reg),
                    X=X,
                    y=y,
                    seeds=seeds,
                    reward_scheme=reward_scheme,
                    progress=progress,
                )
                result['feature_set'] = feature_set_name
                result['alpha'] = alpha
                result['lambda_reg'] = lambda_reg
                result['n_features'] = n_features
                all_results.append(result)

                summary = summarize_results(result)
                summary['feature_set'] = feature_set_name
                summary['alpha'] = alpha
                summary['lambda_reg'] = lambda_reg
                summary['n_features'] = n_features
                all_summaries.append(summary)

    summary_df = pd.concat(all_summaries, ignore_index=True).sort_values('final_accuracy_mean', ascending=False)
    results_df = pd.concat(all_results, ignore_index=True)
    scale_stats_df = pd.concat(all_scale_stats, ignore_index=True) if all_scale_stats else pd.DataFrame()
    return summary_df, results_df, scale_stats_df


def evaluate_feature_set_grid_for_rewards(
    df,
    feature_sets,
    linucb_cls,
    reward_schemes=None,
    **kwargs,
):
    reward_schemes = list(reward_schemes or REWARD_SCHEMES_TO_COMPARE)
    summaries, results, scale_stats = [], [], []
    for reward_scheme in reward_schemes:
        summary, result, scale = evaluate_feature_set_grid(
            df=df,
            feature_sets=feature_sets,
            linucb_cls=linucb_cls,
            reward_scheme=reward_scheme,
            **kwargs,
        )
        summaries.append(summary)
        results.append(result)
        scale_stats.append(scale)
    return (
        pd.concat(summaries, ignore_index=True).sort_values('final_accuracy_mean', ascending=False),
        pd.concat(results, ignore_index=True),
        pd.concat(scale_stats, ignore_index=True) if scale_stats else pd.DataFrame(),
    )



def short_float(value):
    return f'{float(value):g}'


def hybrid_config_name(params):
    return (
        'HybridRidgeLinUCB('
        f"alpha={short_float(params.get('alpha'))}, "
        f"lambda_arm={short_float(params.get('lambda_arm'))}, "
        f"lambda_shared={short_float(params.get('lambda_shared'))}, "
        f"alpha_shared={short_float(params.get('alpha_shared'))}, "
        f"shared_weight={short_float(params.get('shared_weight', 1.0))}"
        ')'
    )


def make_hybrid_linucb_factory(hybrid_cls, n_arms, n_features, params):
    return lambda: hybrid_cls(
        alpha=params.get('alpha', 0.01),
        n_arms=n_arms,
        n_features=n_features,
        lambda_arm=params.get('lambda_arm', 10.0),
        lambda_shared=params.get('lambda_shared', 10.0),
        alpha_shared=params.get('alpha_shared', params.get('alpha', 0.01)),
        shared_weight=params.get('shared_weight', 1.0),
    )


def evaluate_hybrid_feature_set_grid(
    df,
    feature_sets,
    hybrid_cls,
    feature_set_names=None,
    param_grid=None,
    seeds=range(5),
    reward_scheme='binary',
    standardize=True,
    progress=False,
    verbose=True,
):
    feature_set_names = available_final_feature_sets(feature_sets, feature_set_names)
    param_grid = list(param_grid or FINAL_HYBRID_GRID)
    seeds = list(seeds)
    all_summaries, all_results, all_scale_stats = [], [], []

    for feature_set_name in feature_set_names:
        X, y, feature_cols, scale_stats = make_feature_matrix(
            df, feature_sets, feature_set_name, standardize=standardize
        )
        n_arms, n_features = int(len(np.unique(y))), int(X.shape[1])
        scale_stats['feature_set'] = feature_set_name
        scale_stats['reward_scheme'] = str(reward_scheme)
        scale_stats['algorithm_family'] = 'Hybrid RidgeLinUCB'
        all_scale_stats.append(scale_stats)

        if verbose:
            print(f'Hybrid RidgeLinUCB feature set: {feature_set_name} ({n_features} features), reward={reward_scheme}')

        for params in param_grid:
            algorithm = hybrid_config_name(params)
            if verbose:
                print(f'  {algorithm}')
            result = evaluate_bandit(
                name=algorithm,
                bandit_factory=make_hybrid_linucb_factory(hybrid_cls, n_arms, n_features, params),
                X=X,
                y=y,
                seeds=seeds,
                reward_scheme=reward_scheme,
                progress=progress,
            )
            result['feature_set'] = feature_set_name
            result['algorithm_family'] = 'Hybrid RidgeLinUCB'
            result['config_name'] = algorithm
            result['n_features'] = n_features
            for key, value in params.items():
                result[key] = value
            all_results.append(result)

            summary = summarize_results(result)
            summary['feature_set'] = feature_set_name
            summary['algorithm_family'] = 'Hybrid RidgeLinUCB'
            summary['config_name'] = algorithm
            summary['n_features'] = n_features
            for key, value in params.items():
                summary[key] = value
            all_summaries.append(summary)

    summary_df = pd.concat(all_summaries, ignore_index=True).sort_values('final_accuracy_mean', ascending=False)
    results_df = pd.concat(all_results, ignore_index=True)
    scale_stats_df = pd.concat(all_scale_stats, ignore_index=True) if all_scale_stats else pd.DataFrame()
    return summary_df, results_df, scale_stats_df


def evaluate_hybrid_feature_set_grid_for_rewards(
    df,
    feature_sets,
    hybrid_cls,
    reward_schemes=None,
    **kwargs,
):
    reward_schemes = list(reward_schemes or REWARD_SCHEMES_TO_COMPARE)
    summaries, results, scale_stats = [], [], []
    for reward_scheme in reward_schemes:
        summary, result, scale = evaluate_hybrid_feature_set_grid(
            df=df,
            feature_sets=feature_sets,
            hybrid_cls=hybrid_cls,
            reward_scheme=reward_scheme,
            **kwargs,
        )
        summaries.append(summary)
        results.append(result)
        scale_stats.append(scale)
    return (
        pd.concat(summaries, ignore_index=True).sort_values('final_accuracy_mean', ascending=False),
        pd.concat(results, ignore_index=True),
        pd.concat(scale_stats, ignore_index=True) if scale_stats else pd.DataFrame(),
    )


def add_error_columns(results):
    out = results.copy()
    out['absolute_error'] = (out['chosen_arm'] - out['true_arm']).abs()
    out['severe_error'] = (out['absolute_error'] == 2).astype(int)
    out['one_step_error'] = (out['absolute_error'] == 1).astype(int)
    out['underdose_error'] = (out['chosen_arm'] < out['true_arm']).astype(int)
    out['overdose_error'] = (out['chosen_arm'] > out['true_arm']).astype(int)
    if 'is_correct' not in out.columns:
        out['is_correct'] = (out['chosen_arm'] == out['true_arm']).astype(int)
    return out


def _metric_group_cols(frame):
    candidates = [
        'feature_set', 'algorithm_family', 'algorithm', 'config_name', 'reward_scheme',
        'alpha', 'lambda_reg', 'lambda_arm', 'lambda_shared', 'alpha_shared', 'shared_weight',
        'sample_scale', 'covariance_type',
        'lasso_alpha', 'ucb_alpha', 'min_samples_per_arm', 'refit_frequency', 'run',
    ]
    return [col for col in candidates if col in frame.columns]


def summarize_error_metrics(results):
    out = add_error_columns(results)
    keys = _metric_group_cols(out)
    by_run = out.groupby(keys, dropna=False).agg(
        accuracy=('is_correct', 'mean'),
        mean_reward=('reward', 'mean'),
        error_rate=('is_correct', lambda s: 1.0 - float(np.mean(s))),
        mean_regret=('regret', 'mean'),
        severe_error_rate=('severe_error', 'mean'),
        one_step_error_rate=('one_step_error', 'mean'),
        underdose_rate=('underdose_error', 'mean'),
        overdose_rate=('overdose_error', 'mean'),
    ).reset_index()

    group_keys = [col for col in keys if col != 'run']
    summary = by_run.groupby(group_keys, dropna=False).agg(
        n_runs=('run', 'nunique'),
        accuracy_mean=('accuracy', 'mean'),
        accuracy_std=('accuracy', 'std'),
        mean_reward_mean=('mean_reward', 'mean'),
        mean_reward_std=('mean_reward', 'std'),
        mean_regret_mean=('mean_regret', 'mean'),
        mean_regret_std=('mean_regret', 'std'),
        severe_error_rate_mean=('severe_error_rate', 'mean'),
        severe_error_rate_std=('severe_error_rate', 'std'),
        one_step_error_rate_mean=('one_step_error_rate', 'mean'),
        underdose_rate_mean=('underdose_rate', 'mean'),
        overdose_rate_mean=('overdose_rate', 'mean'),
    ).reset_index()
    summary['accuracy_ci95'] = 1.96 * summary['accuracy_std'].fillna(0) / np.sqrt(summary['n_runs'])
    summary['mean_reward_ci95'] = 1.96 * summary['mean_reward_std'].fillna(0) / np.sqrt(summary['n_runs'])
    summary['severe_error_rate_ci95'] = 1.96 * summary['severe_error_rate_std'].fillna(0) / np.sqrt(summary['n_runs'])
    return summary.sort_values('accuracy_mean', ascending=False)


def classwise_metrics(results, labels=(0, 1, 2), label_names=ARM_LABELS):
    rows = []
    group_cols = _metric_group_cols(results)

    for key, group in results.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_cols, key))
        y_true = group['true_arm'].astype(int).to_numpy()
        y_pred = group['chosen_arm'].astype(int).to_numpy()
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(labels), zero_division=0
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
    summary = by_run.groupby(summary_keys, dropna=False).agg(
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
    group_cols = [col for col in _metric_group_cols(results) if col != 'run']

    for key, group in results.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
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


def select_top_feature_sets(summary, top_n=2):
    best = summary.sort_values('final_accuracy_mean', ascending=False)
    return best.drop_duplicates('feature_set').head(top_n)['feature_set'].tolist()


def save_benchmark_outputs(output_dir, save_step_results=False, **tables):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, table in tables.items():
        if table is None or len(table) == 0:
            continue
        if not save_step_results and name.endswith(('_results', '_step_results')):
            continue
        path = output_dir / f'{name}.csv'
        table.to_csv(path, index=False)
        written[name] = str(path)
    with open(output_dir / 'benchmark_outputs.json', 'w') as f:
        json.dump(written, f, indent=2)
    return written


def plot_feature_set_accuracy(summary, top_n=None, title='RidgeLinUCB feature-set comparison'):
    plot_df = summary.sort_values('final_accuracy_mean', ascending=False).copy()
    if top_n:
        plot_df = plot_df.head(top_n)

    if 'config_name' in plot_df.columns:
        family = plot_df['algorithm_family'].fillna('RidgeLinUCB') if 'algorithm_family' in plot_df.columns else 'RidgeLinUCB'
        labels = family.astype(str) + '\n' + plot_df['feature_set'].astype(str) + '\n' + plot_df['config_name'].astype(str)
    else:
        labels = plot_df['feature_set'] + '\nα=' + plot_df['alpha'].astype(str) + ', λ=' + plot_df['lambda_reg'].astype(str)
    if 'reward_scheme' in plot_df.columns:
        labels = labels + '\nreward=' + plot_df['reward_scheme'].astype(str)

    fig, ax = plt.subplots(figsize=(12, max(5, 0.55 * len(plot_df))))
    ax.barh(np.arange(len(plot_df)), plot_df['final_accuracy_mean'])
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Mean final accuracy')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.25)
    plt.tight_layout()
    return fig, ax
