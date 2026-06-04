import inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


REWARD_SCHEMES = {
    'binary': '1 for correct bucket, 0 otherwise',
    'ordinal': '1 for correct bucket, 0 for one-bucket error, -1 for low/high severe error',
}


def _method_accepts_x(method):
    return len(inspect.signature(method).parameters) >= 1


def _update_accepts_x(method):
    return len(inspect.signature(method).parameters) >= 3


def compute_reward(chosen_arm, true_arm, reward_scheme='binary'):
    """
    Return the scalar reward used to update the bandit.

    binary:
        r = 1 if chosen bucket equals true bucket, else 0.
    ordinal:
        r = 1 for exact bucket, 0 for adjacent bucket error, -1 for severe low/high error.

    A callable can also be passed for future experiments. It should accept
    (chosen_arm, true_arm) and return a scalar reward.
    """
    chosen_arm = int(chosen_arm)
    true_arm = int(true_arm)

    if callable(reward_scheme):
        return float(reward_scheme(chosen_arm, true_arm))

    if reward_scheme == 'binary':
        return 1.0 if chosen_arm == true_arm else 0.0
    if reward_scheme == 'ordinal':
        distance = abs(chosen_arm - true_arm)
        if distance == 0:
            return 1.0
        if distance == 1:
            return 0.0
        return -1.0

    raise ValueError(
        f"Unknown reward_scheme={reward_scheme!r}. Supported values: {list(REWARD_SCHEMES)} or a callable."
    )


def reward_scheme_name(reward_scheme):
    return getattr(reward_scheme, '__name__', 'custom') if callable(reward_scheme) else str(reward_scheme)


def run_bandit_once(
    bandit,
    X,
    y,
    seed=0,
    run_id=0,
    name='bandit',
    reward_scheme='binary',
    optimal_reward=1.0,
    progress=False,
):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    select_uses_x = _method_accepts_x(bandit.select_arm)
    update_uses_x = _update_accepts_x(bandit.update)
    reward_label = reward_scheme_name(reward_scheme)

    records = []
    iterator = tqdm(order, desc=f'{name} run {run_id}', leave=False) if progress else order

    cumulative_regret = 0.0
    cumulative_reward = 0.0
    correct = 0
    mistakes = 0

    for t, idx in enumerate(iterator, start=1):
        x = X[idx]
        actual_arm = int(y[idx])
        chosen_arm = int(bandit.select_arm(x)) if select_uses_x else int(bandit.select_arm())
        is_correct = int(chosen_arm == actual_arm)
        reward = float(compute_reward(chosen_arm, actual_arm, reward_scheme=reward_scheme))

        if update_uses_x:
            bandit.update(chosen_arm, x, reward)
        else:
            bandit.update(chosen_arm, reward)

        correct += is_correct
        mistakes += int(not is_correct)
        cumulative_reward += reward
        regret = float(optimal_reward - reward)
        cumulative_regret += regret

        records.append({
            'algorithm': name,
            'reward_scheme': reward_label,
            'run': run_id,
            'seed': seed,
            't': t,
            'row_index': int(idx),
            'chosen_arm': chosen_arm,
            'true_arm': actual_arm,
            'is_correct': is_correct,
            'reward': reward,
            'regret': regret,
            'cumulative_reward': cumulative_reward,
            'cumulative_mean_reward': cumulative_reward / t,
            'cumulative_regret': cumulative_regret,
            'cumulative_fraction_incorrect': mistakes / t,
            'cumulative_accuracy': correct / t,
        })

    return pd.DataFrame(records)


def _make_bandit_from_factory(bandit_factory, seed, run_id):
    signature = inspect.signature(bandit_factory)
    kwargs = {}
    if 'seed' in signature.parameters:
        kwargs['seed'] = seed
    if 'run_id' in signature.parameters:
        kwargs['run_id'] = run_id
    return bandit_factory(**kwargs) if kwargs else bandit_factory()


def evaluate_bandit(
    name,
    bandit_factory,
    X,
    y,
    seeds=None,
    reward_scheme='binary',
    progress=False,
):
    if seeds is None:
        seeds = range(20)

    runs = []
    for run_id, seed in enumerate(seeds):
        bandit = _make_bandit_from_factory(bandit_factory, seed, run_id)
        runs.append(
            run_bandit_once(
                bandit=bandit,
                X=X,
                y=y,
                seed=seed,
                run_id=run_id,
                name=name,
                reward_scheme=reward_scheme,
                progress=progress,
            )
        )
    return pd.concat(runs, ignore_index=True)


def evaluate_static_predictions(name, y_true, y_pred, seeds=None, reward_scheme='binary'):
    if seeds is None:
        seeds = range(20)

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    reward_label = reward_scheme_name(reward_scheme)
    rows = []

    for run_id, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(y_true))
        cumulative_regret = 0.0
        cumulative_reward = 0.0
        correct = 0
        mistakes = 0
        for t, idx in enumerate(order, start=1):
            chosen_arm = int(y_pred[idx])
            actual_arm = int(y_true[idx])
            is_correct = int(chosen_arm == actual_arm)
            reward = float(compute_reward(chosen_arm, actual_arm, reward_scheme=reward_scheme))
            correct += is_correct
            mistakes += int(not is_correct)
            cumulative_reward += reward
            regret = 1.0 - reward
            cumulative_regret += regret
            rows.append({
                'algorithm': name,
                'reward_scheme': reward_label,
                'run': run_id,
                'seed': seed,
                't': t,
                'row_index': int(idx),
                'chosen_arm': chosen_arm,
                'true_arm': actual_arm,
                'is_correct': is_correct,
                'reward': reward,
                'regret': regret,
                'cumulative_reward': cumulative_reward,
                'cumulative_mean_reward': cumulative_reward / t,
                'cumulative_regret': cumulative_regret,
                'cumulative_fraction_incorrect': mistakes / t,
                'cumulative_accuracy': correct / t,
            })

    return pd.DataFrame(rows)


def summarize_results(results, late_window=500):
    metric_cols = ['algorithm', 'run']
    if 'reward_scheme' in results.columns:
        metric_cols.insert(1, 'reward_scheme')

    final_rows = results.sort_values('t').groupby(metric_cols, dropna=False).tail(1)
    late_rows = results.groupby(metric_cols, dropna=False).tail(late_window)
    late_accuracy_source = 'is_correct' if 'is_correct' in late_rows.columns else 'reward'

    late_accuracy = (
        late_rows.groupby(metric_cols, dropna=False)[late_accuracy_source]
        .mean()
        .reset_index(name='late_accuracy')
    )
    late_reward = (
        late_rows.groupby(metric_cols, dropna=False)['reward']
        .mean()
        .reset_index(name='late_mean_reward')
    )
    merged = final_rows.merge(late_accuracy, on=metric_cols, how='left').merge(
        late_reward, on=metric_cols, how='left'
    )

    group_cols = ['algorithm']
    if 'reward_scheme' in merged.columns:
        group_cols.append('reward_scheme')

    summary = merged.groupby(group_cols, dropna=False).agg(
        n_runs=('run', 'nunique'),
        final_accuracy_mean=('cumulative_accuracy', 'mean'),
        final_accuracy_std=('cumulative_accuracy', 'std'),
        final_reward_mean=('cumulative_mean_reward', 'mean'),
        final_reward_std=('cumulative_mean_reward', 'std'),
        final_regret_mean=('cumulative_regret', 'mean'),
        final_regret_std=('cumulative_regret', 'std'),
        final_fraction_incorrect_mean=('cumulative_fraction_incorrect', 'mean'),
        final_fraction_incorrect_std=('cumulative_fraction_incorrect', 'std'),
        late_accuracy_mean=('late_accuracy', 'mean'),
        late_accuracy_std=('late_accuracy', 'std'),
        late_mean_reward_mean=('late_mean_reward', 'mean'),
        late_mean_reward_std=('late_mean_reward', 'std'),
    ).reset_index()

    for col in [
        'final_accuracy', 'final_reward', 'final_regret', 'final_fraction_incorrect',
        'late_accuracy', 'late_mean_reward',
    ]:
        std_col = f'{col}_std'
        mean_col = f'{col}_mean'
        ci_col = f'{col}_ci95'
        summary[ci_col] = 1.96 * summary[std_col].fillna(0) / np.sqrt(summary['n_runs'])
        summary[mean_col] = summary[mean_col].astype(float)

    return summary.sort_values('final_accuracy_mean', ascending=False)


def learning_curve_summary(results, metric='cumulative_fraction_incorrect'):
    group_cols = ['algorithm', 't']
    if 'reward_scheme' in results.columns:
        group_cols.insert(1, 'reward_scheme')
    grouped = results.groupby(group_cols, dropna=False)[metric]
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

    group_cols = ['algorithm']
    if 'reward_scheme' in curve.columns:
        group_cols.append('reward_scheme')

    for key, group in curve.groupby(group_cols, dropna=False):
        group = group.sort_values('t')
        if max_points and len(group) > max_points:
            step = int(np.ceil(len(group) / max_points))
            group = group.iloc[::step]
        label = key if isinstance(key, str) else ' | '.join(map(str, key))
        x = group['t'].to_numpy(dtype=float)
        y = group['mean'].to_numpy(dtype=float)
        ax.plot(x, y, label=label)
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
    reward_scheme='binary',
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
                reward_scheme=reward_scheme,
                progress=progress,
            )
            summary = summarize_results(result)
            summary['alpha'] = alpha
            summary['lambda_reg'] = lambda_reg
            all_results.append(summary)

    return pd.concat(all_results, ignore_index=True).sort_values('final_accuracy_mean', ascending=False)
