from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.bandit_evaluation import evaluate_bandit, summarize_results
from src.evaluation.feature_set_benchmark import (
    classwise_metrics,
    confusion_table,
    make_feature_matrix,
    save_benchmark_outputs,
    static_policy_summary,
    summarize_error_metrics,
)


DEFAULT_TS_GRID = [
    {'lambda_reg': 5.0, 'sample_scale': 0.01, 'covariance_type': 'diagonal'},
    {'lambda_reg': 10.0, 'sample_scale': 0.01, 'covariance_type': 'diagonal'},
    {'lambda_reg': 10.0, 'sample_scale': 0.025, 'covariance_type': 'diagonal'},
    {'lambda_reg': 20.0, 'sample_scale': 0.025, 'covariance_type': 'diagonal'},
    {'lambda_reg': 10.0, 'sample_scale': 0.05, 'covariance_type': 'diagonal'},
]

DEFAULT_LASSO_GRID = [
    {'lasso_alpha': 0.0005, 'ucb_alpha': 0.0, 'lambda_reg': 10.0, 'min_samples_per_arm': 15, 'refit_frequency': 50},
    {'lasso_alpha': 0.0010, 'ucb_alpha': 0.0, 'lambda_reg': 10.0, 'min_samples_per_arm': 15, 'refit_frequency': 50},
    {'lasso_alpha': 0.0025, 'ucb_alpha': 0.0, 'lambda_reg': 10.0, 'min_samples_per_arm': 20, 'refit_frequency': 50},
    {'lasso_alpha': 0.0010, 'ucb_alpha': 0.025, 'lambda_reg': 10.0, 'min_samples_per_arm': 20, 'refit_frequency': 50},
    {'lasso_alpha': 0.0025, 'ucb_alpha': 0.025, 'lambda_reg': 10.0, 'min_samples_per_arm': 20, 'refit_frequency': 50},
]


def short_float(value):
    return f'{float(value):g}'


def get_intercept_index(feature_cols, intercept_name='Intercept'):
    return feature_cols.index(intercept_name) if intercept_name in feature_cols else None


def ts_config_name(params):
    return (
        'LinearTS('
        f"lambda={short_float(params.get('lambda_reg'))}, "
        f"scale={short_float(params.get('sample_scale'))}, "
        f"cov={params.get('covariance_type', 'diagonal')}"
        ')'
    )


def lasso_config_name(params):
    return (
        'LassoLinUCB('
        f"lasso={short_float(params.get('lasso_alpha'))}, "
        f"alpha={short_float(params.get('ucb_alpha'))}, "
        f"lambda={short_float(params.get('lambda_reg'))}, "
        f"warmup={params.get('min_samples_per_arm')}, "
        f"refit={params.get('refit_frequency')}"
        ')'
    )


def make_linear_ts_factory(bandit_cls, n_arms, n_features, params):
    def factory(seed=None, run_id=None):
        return bandit_cls(
            n_arms=n_arms,
            n_features=n_features,
            lambda_reg=params.get('lambda_reg', 10.0),
            sample_scale=params.get('sample_scale', 0.025),
            covariance_type=params.get('covariance_type', 'diagonal'),
            random_state=seed,
        )
    return factory


def make_lasso_factory(bandit_cls, n_arms, n_features, params, intercept_index=None):
    def factory(seed=None, run_id=None):
        return bandit_cls(
            n_arms=n_arms,
            n_features=n_features,
            lasso_alpha=params.get('lasso_alpha', 0.001),
            ucb_alpha=params.get('ucb_alpha', 0.0),
            lambda_reg=params.get('lambda_reg', 10.0),
            min_samples_per_arm=params.get('min_samples_per_arm', 20),
            refit_frequency=params.get('refit_frequency', 50),
            max_iter=params.get('max_iter', 2500),
            tol=params.get('tol', 1e-4),
            coefficient_tol=params.get('coefficient_tol', 1e-8),
            intercept_index=intercept_index,
            random_state=seed,
        )
    return factory


def _attach_common_metadata(result, feature_set_name, params, n_features, algorithm_family, config_name):
    out = result.copy()
    out['feature_set'] = feature_set_name
    out['algorithm_family'] = algorithm_family
    out['config_name'] = config_name
    out['n_features'] = int(n_features)
    for key, value in params.items():
        out[key] = value
    return out


def _summarize_with_metadata(result):
    meta_cols = [
        'feature_set', 'algorithm_family', 'config_name', 'n_features',
        'lambda_reg', 'sample_scale', 'covariance_type',
        'lasso_alpha', 'ucb_alpha', 'min_samples_per_arm', 'refit_frequency',
    ]
    meta = {col: result[col].iloc[0] for col in meta_cols if col in result.columns}
    summary = summarize_results(result)
    for col, value in meta.items():
        summary[col] = value
    return summary


def evaluate_linear_ts_grid(
    df,
    feature_sets,
    bandit_cls,
    feature_set_names,
    param_grid=None,
    seeds=range(3),
    standardize=True,
    progress=False,
    verbose=True,
):
    param_grid = list(param_grid or DEFAULT_TS_GRID)
    all_results = []
    all_summaries = []
    all_scale_stats = []

    for feature_set_name in feature_set_names:
        X, y, feature_cols, scale_stats = make_feature_matrix(
            df, feature_sets, feature_set_name, standardize=standardize
        )
        n_arms = int(len(np.unique(y)))
        n_features = int(X.shape[1])
        scale_stats['feature_set'] = feature_set_name
        all_scale_stats.append(scale_stats)

        if verbose:
            print(f'LinearTS feature set: {feature_set_name} ({n_features} features)')

        for params in param_grid:
            config_name = ts_config_name(params)
            if verbose:
                print(f'  {config_name}')
            result = evaluate_bandit(
                name=config_name,
                bandit_factory=make_linear_ts_factory(bandit_cls, n_arms, n_features, params),
                X=X,
                y=y,
                seeds=seeds,
                progress=progress,
            )
            result = _attach_common_metadata(
                result, feature_set_name, params, n_features, 'Linear Thompson Sampling', config_name
            )
            all_results.append(result)
            all_summaries.append(_summarize_with_metadata(result))

    results_df = pd.concat(all_results, ignore_index=True)
    summary_df = pd.concat(all_summaries, ignore_index=True).sort_values(
        'final_accuracy_mean', ascending=False
    )
    scale_stats_df = pd.concat(all_scale_stats, ignore_index=True) if all_scale_stats else pd.DataFrame()
    return summary_df, results_df, scale_stats_df


def evaluate_lasso_grid(
    df,
    feature_sets,
    bandit_cls,
    feature_set_names,
    param_grid=None,
    seeds=range(3),
    standardize=True,
    progress=False,
    verbose=True,
):
    param_grid = list(param_grid or DEFAULT_LASSO_GRID)
    all_results = []
    all_summaries = []
    all_scale_stats = []

    for feature_set_name in feature_set_names:
        X, y, feature_cols, scale_stats = make_feature_matrix(
            df, feature_sets, feature_set_name, standardize=standardize
        )
        n_arms = int(len(np.unique(y)))
        n_features = int(X.shape[1])
        intercept_index = get_intercept_index(feature_cols)
        scale_stats['feature_set'] = feature_set_name
        all_scale_stats.append(scale_stats)

        if verbose:
            print(f'LassoLinUCB feature set: {feature_set_name} ({n_features} features)')

        for params in param_grid:
            config_name = lasso_config_name(params)
            if verbose:
                print(f'  {config_name}')
            result = evaluate_bandit(
                name=config_name,
                bandit_factory=make_lasso_factory(
                    bandit_cls, n_arms, n_features, params, intercept_index=intercept_index
                ),
                X=X,
                y=y,
                seeds=seeds,
                progress=progress,
            )
            result = _attach_common_metadata(
                result, feature_set_name, params, n_features, 'LassoLinUCB', config_name
            )
            result['intercept_index'] = -1 if intercept_index is None else int(intercept_index)
            all_results.append(result)
            all_summaries.append(_summarize_with_metadata(result))

    results_df = pd.concat(all_results, ignore_index=True)
    summary_df = pd.concat(all_summaries, ignore_index=True).sort_values(
        'final_accuracy_mean', ascending=False
    )
    scale_stats_df = pd.concat(all_scale_stats, ignore_index=True) if all_scale_stats else pd.DataFrame()
    return summary_df, results_df, scale_stats_df


def combine_advanced_results(*result_frames):
    frames = [frame for frame in result_frames if frame is not None and len(frame) > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_advanced_errors(results):
    return summarize_error_metrics(results)


def classwise_advanced_metrics(results):
    return classwise_metrics(results)


def advanced_confusion_table(results, normalize='true'):
    return confusion_table(results, normalize=normalize)


def save_advanced_benchmark(output_dir, prefix, summary, results, scale_stats=None):
    output_dir = Path(output_dir)
    error_summary = summarize_advanced_errors(results)
    classwise = classwise_advanced_metrics(results)
    confusion_true = advanced_confusion_table(results, normalize='true')
    confusion_counts = advanced_confusion_table(results, normalize=None)

    return save_benchmark_outputs(
        output_dir,
        **{
            f'{prefix}_summary': summary,
            f'{prefix}_results': results,
            f'{prefix}_error_summary': error_summary,
            f'{prefix}_classwise': classwise,
            f'{prefix}_confusion_true_normalized': confusion_true,
            f'{prefix}_confusion_counts': confusion_counts,
            f'{prefix}_scale_stats': scale_stats,
        },
    )


def load_top_feature_sets_from_ridge_results(path, top_n=3):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Missing RidgeLinUCB leaderboard: {path}')
    leaderboard = pd.read_csv(path).sort_values('final_accuracy_mean', ascending=False)
    return leaderboard.drop_duplicates('feature_set').head(top_n)['feature_set'].tolist()


def plot_advanced_leaderboard(summary, top_n=20, title='Advanced contextual bandit leaderboard'):
    plot_df = summary.sort_values('final_accuracy_mean', ascending=False).head(top_n).copy()
    labels = plot_df['algorithm_family'] + '\n' + plot_df['feature_set'] + '\n' + plot_df['config_name']

    fig, ax = plt.subplots(figsize=(13, max(6, 0.6 * len(plot_df))))
    ax.barh(np.arange(len(plot_df)), plot_df['final_accuracy_mean'])
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Mean final accuracy')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.25)
    plt.tight_layout()
    return fig, ax


def make_static_baselines_for_feature_sets(df, feature_set_names):
    return static_policy_summary(df, feature_set_names)
