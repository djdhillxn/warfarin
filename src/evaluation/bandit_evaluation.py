import inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def _method_accepts_x(method):
    return len(inspect.signature(method).parameters) >= 1


def _update_accepts_x(method):
    return len(inspect.signature(method).parameters) >= 3


def run_bandit_once(
    bandit,
    X,
    y,
    seed=0,
    run_id=0,
    name='bandit',
    progress=False,
):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    select_uses_x = _method_accepts_x(bandit.select_arm)
    update_uses_x = _update_accepts_x(bandit.update)

    records = []
    iterator = tqdm(order, desc=f'{name} run {run_id}', leave=False) if progress else order

    cumulative_regret = 0
    correct = 0
    for t, idx in enumerate(iterator, start=1):
        x = X[idx]
        actual_arm = int(y[idx])
        chosen_arm = int(bandit.select_arm(x)) if select_uses_x else int(bandit.select_arm())
        reward = 1 if chosen_arm == actual_arm else 0

        if update_uses_x:
            bandit.update(chosen_arm, x, reward)
        else:
            bandit.update(chosen_arm, reward)

        correct += reward
        regret = 1 - reward
        cumulative_regret += regret

        records.append({
            'algorithm': name,
            'run': run_id,
            'seed': seed,
            't': t,
            'row_index': int(idx),
            'chosen_arm': chosen_arm,
            'true_arm': actual_arm,
            'reward': reward,
            'regret': regret,
            'cumulative_regret': cumulative_regret,
            'cumulative_fraction_incorrect': cumulative_regret / t,
            'cumulative_accuracy': correct / t,
        })

    return pd.DataFrame(records)


def evaluate_bandit(
    name,
    bandit_factory,
    X,
    y,
    seeds=None,
    progress=False,
):
    if seeds is None:
        seeds = range(20)

    runs = []
    for run_id, seed in enumerate(seeds):
        bandit = bandit_factory()
        runs.append(
            run_bandit_once(
                bandit=bandit,
                X=X,
                y=y,
                seed=seed,
                run_id=run_id,
                name=name,
                progress=progress,
            )
        )
    return pd.concat(runs, ignore_index=True)


def evaluate_static_predictions(name, y_true, y_pred, seeds=None):
    if seeds is None:
        seeds = range(20)

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    rows = []

    for run_id, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(y_true))
        cumulative_regret = 0
        correct = 0
        for t, idx in enumerate(order, start=1):
            chosen_arm = int(y_pred[idx])
            actual_arm = int(y_true[idx])
            reward = 1 if chosen_arm == actual_arm else 0
            correct += reward
            regret = 1 - reward
            cumulative_regret += regret
            rows.append({
                'algorithm': name,
                'run': run_id,
                'seed': seed,
                't': t,
                'row_index': int(idx),
                'chosen_arm': chosen_arm,
                'true_arm': actual_arm,
                'reward': reward,
                'regret': regret,
                'cumulative_regret': cumulative_regret,
                'cumulative_fraction_incorrect': cumulative_regret / t,
                'cumulative_accuracy': correct / t,
            })

    return pd.DataFrame(rows)


def summarize_results(results, late_window=500):
    final_rows = results.sort_values('t').groupby(['algorithm', 'run']).tail(1)
    late_rows = results.groupby(['algorithm', 'run']).tail(late_window)
    late_accuracy = late_rows.groupby(['algorithm', 'run'])['reward'].mean().reset_index(name='late_accuracy')
    merged = final_rows.merge(late_accuracy, on=['algorithm', 'run'], how='left')

    summary = merged.groupby('algorithm').agg(
        n_runs=('run', 'nunique'),
        final_accuracy_mean=('cumulative_accuracy', 'mean'),
        final_accuracy_std=('cumulative_accuracy', 'std'),
        final_regret_mean=('cumulative_regret', 'mean'),
        final_regret_std=('cumulative_regret', 'std'),
        final_fraction_incorrect_mean=('cumulative_fraction_incorrect', 'mean'),
        final_fraction_incorrect_std=('cumulative_fraction_incorrect', 'std'),
        late_accuracy_mean=('late_accuracy', 'mean'),
        late_accuracy_std=('late_accuracy', 'std'),
    ).reset_index()

    for col in ['final_accuracy', 'final_regret', 'final_fraction_incorrect', 'late_accuracy']:
        std_col = f'{col}_std'
        mean_col = f'{col}_mean'
        ci_col = f'{col}_ci95'
        summary[ci_col] = 1.96 * summary[std_col].fillna(0) / np.sqrt(summary['n_runs'])
        summary[mean_col] = summary[mean_col].astype(float)

    return summary.sort_values('final_accuracy_mean', ascending=False)


def learning_curve_summary(results, metric='cumulative_fraction_incorrect'):
    grouped = results.groupby(['algorithm', 't'])[metric]
    summary = grouped.agg(['mean', 'std', 'count']).reset_index()
    summary['ci95'] = 1.96 * summary['std'].fillna(0) / np.sqrt(summary['count'])
    return summary


def plot_learning_curves(
    results,
    metric='cumulative_fraction_incorrect',
    baseline_rates=None,
    title=None,
    figsize=(11, 6),
    ci=True,
    max_points=700,
):
    curve = learning_curve_summary(results, metric=metric)
    fig, ax = plt.subplots(figsize=figsize)

    for algorithm, group in curve.groupby('algorithm'):
        group = group.sort_values('t')
        if max_points and len(group) > max_points:
            step = int(np.ceil(len(group) / max_points))
            group = group.iloc[::step]
        x = group['t'].to_numpy(dtype=float)
        y = group['mean'].to_numpy(dtype=float)
        ax.plot(x, y, label=algorithm)
        if ci:
            low = y - group['ci95'].to_numpy(dtype=float)
            high = y + group['ci95'].to_numpy(dtype=float)
            ax.fill_between(x, low, high, alpha=0.18)

    if baseline_rates:
        t_max = int(results['t'].max())
        x = np.arange(1, t_max + 1)
        for label, rate in baseline_rates.items():
            if metric == 'cumulative_regret':
                ax.plot(x, rate * x, linestyle='--', label=label)
            elif metric == 'cumulative_fraction_incorrect':
                ax.axhline(rate, linestyle='--', label=label)

    if title is None:
        title = metric.replace('_', ' ').title()
    ax.set_title(title)
    ax.set_xlabel('Patients seen')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig, ax


def tune_linucb_grid(
    linucb_cls,
    X,
    y,
    alphas,
    lambda_regs,
    seeds=None,
    progress=False,
):
    if seeds is None:
        seeds = range(5)

    n_arms = int(len(np.unique(y)))
    n_features = int(X.shape[1])
    all_results = []

    for alpha in alphas:
        for lambda_reg in lambda_regs:
            name = f'LinUCB alpha={alpha}, lambda={lambda_reg}'
            result = evaluate_bandit(
                name=name,
                bandit_factory=lambda a=alpha, l=lambda_reg: linucb_cls(
                    alpha=a,
                    n_arms=n_arms,
                    n_features=n_features,
                    lambda_reg=l,
                ),
                X=X,
                y=y,
                seeds=seeds,
                progress=progress,
            )
            summary = summarize_results(result)
            summary['alpha'] = alpha
            summary['lambda_reg'] = lambda_reg
            all_results.append(summary)

    return pd.concat(all_results, ignore_index=True).sort_values(
        'final_accuracy_mean', ascending=False
    )
