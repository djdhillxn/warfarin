import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.bandit_evaluation import compute_reward
from src.evaluation.feature_set_benchmark import ARM_LABELS, make_feature_matrix, save_benchmark_outputs


GENOTYPE_KEYWORDS = ['cyp2c9', 'vkorc1', 'rs9923231', 'allele']
RACE_KEYWORDS = ['race_']
GENDER_KEYWORDS = ['gender_']
AGE_KEYWORDS = ['age__', 'age_', 'age in decades']
HEIGHT_WEIGHT_KEYWORDS = ['height', 'weight', 'bsa', 'bmi']
MISSINGNESS_KEYWORDS = ['missing', 'unknown', 'nan']
TARGET_INR_KEYWORDS = ['targetinr', 'target_inr', 'inr']
INTERACTION_KEYWORDS = ['interaction__']
MEDICATION_KEYWORDS = ['medication', 'drug_', 'amiodarone', 'carbamazepine', 'phenytoin', 'rifampin', 'statin', 'antibiotic', 'antiplatelet', 'nsaid']
COMORBIDITY_KEYWORDS = ['comorbidity', 'diabetes', 'heart', 'failure', 'liver', 'renal', 'kidney', 'cancer', 'stroke', 'hypertension']
INDICATION_KEYWORDS = ['indication_']
INTERCEPT_KEYWORDS = ['intercept']


def _method_accepts_x(method):
    return len(inspect.signature(method).parameters) >= 1


def _update_accepts_x(method):
    return len(inspect.signature(method).parameters) >= 3


def fit_bandit_for_audit(
    bandit,
    X,
    y,
    seed=0,
    reward_scheme='binary',
    name='audit_bandit',
):
    """Run one online pass and return the mutated bandit plus per-step records."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    select_uses_x = _method_accepts_x(bandit.select_arm)
    update_uses_x = _update_accepts_x(bandit.update)
    rows = []
    correct = 0
    cumulative_reward = 0.0
    cumulative_regret = 0.0

    for t, idx in enumerate(order, start=1):
        x = X[idx]
        true_arm = int(y[idx])
        chosen_arm = int(bandit.select_arm(x)) if select_uses_x else int(bandit.select_arm())
        reward = float(compute_reward(chosen_arm, true_arm, reward_scheme=reward_scheme))
        if update_uses_x:
            bandit.update(chosen_arm, x, reward)
        else:
            bandit.update(chosen_arm, reward)

        is_correct = int(chosen_arm == true_arm)
        correct += is_correct
        cumulative_reward += reward
        cumulative_regret += 1.0 - reward
        rows.append({
            'algorithm': name,
            'reward_scheme': str(reward_scheme),
            'seed': int(seed),
            't': int(t),
            'row_index': int(idx),
            'chosen_arm': chosen_arm,
            'true_arm': true_arm,
            'is_correct': is_correct,
            'reward': reward,
            'cumulative_accuracy': correct / t,
            'cumulative_reward': cumulative_reward,
            'cumulative_regret': cumulative_regret,
        })

    return bandit, pd.DataFrame(rows)


def get_arm_coefficient_matrix(bandit, coefficient_mode='effective'):
    """
    Return an n_arms x n_features coefficient matrix.

    coefficient_mode='effective' uses HybridRidgeLinUCB.effective_theta() when available,
    so the reported coefficients match the mean-score coefficients used by the policy.
    coefficient_mode='arm_specific' forces the raw per-arm theta matrix.
    """
    if coefficient_mode == 'effective' and hasattr(bandit, 'effective_theta'):
        coef = bandit.effective_theta()
    elif hasattr(bandit, 'theta'):
        coef = bandit.theta
    else:
        raise AttributeError('Bandit does not expose theta or effective_theta coefficients.')

    coef = np.asarray(coef, dtype=float)
    if coef.ndim != 2:
        raise ValueError(f'Expected coefficient matrix with 2 dimensions, got shape {coef.shape}.')
    return coef


def coefficients_long_table(
    bandit,
    feature_cols,
    algorithm,
    feature_set,
    reward_scheme='binary',
    coefficient_mode='effective',
    arm_labels=ARM_LABELS,
):
    coef = get_arm_coefficient_matrix(bandit, coefficient_mode=coefficient_mode)
    rows = []
    for arm in range(coef.shape[0]):
        for idx, feature in enumerate(feature_cols):
            value = float(coef[arm, idx])
            rows.append({
                'algorithm': algorithm,
                'feature_set': feature_set,
                'reward_scheme': str(reward_scheme),
                'coefficient_mode': coefficient_mode,
                'arm_id': int(arm),
                'arm_name': arm_labels.get(int(arm), str(arm)),
                'feature_index': int(idx),
                'feature': feature,
                'coefficient': value,
                'abs_coefficient': abs(value),
                'feature_group': feature_group(feature),
            })
    return pd.DataFrame(rows)


def top_coefficients(coef_long, top_n=25):
    return (
        coef_long.sort_values(['arm_id', 'abs_coefficient'], ascending=[True, False])
        .groupby(['algorithm', 'feature_set', 'reward_scheme', 'coefficient_mode', 'arm_id', 'arm_name'], dropna=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def coefficient_contrasts(coef_long, baseline_arm=1, top_n=30):
    wide = coef_long.pivot_table(
        index=['algorithm', 'feature_set', 'reward_scheme', 'coefficient_mode', 'feature', 'feature_group'],
        columns='arm_id',
        values='coefficient',
        aggfunc='first',
    ).reset_index()

    rows = []
    arm_cols = [col for col in wide.columns if isinstance(col, (int, np.integer))]
    for arm in arm_cols:
        if int(arm) == int(baseline_arm):
            continue
        diff = wide[arm] - wide[baseline_arm]
        tmp = wide[['algorithm', 'feature_set', 'reward_scheme', 'coefficient_mode', 'feature', 'feature_group']].copy()
        tmp['contrast'] = f'{ARM_LABELS.get(int(arm), arm)}_minus_{ARM_LABELS.get(int(baseline_arm), baseline_arm)}'
        tmp['arm_id'] = int(arm)
        tmp['baseline_arm_id'] = int(baseline_arm)
        tmp['coefficient_difference'] = diff.astype(float)
        tmp['abs_coefficient_difference'] = tmp['coefficient_difference'].abs()
        rows.append(tmp)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out) == 0:
        return out
    return (
        out.sort_values(['contrast', 'abs_coefficient_difference'], ascending=[True, False])
        .groupby(['algorithm', 'feature_set', 'reward_scheme', 'coefficient_mode', 'contrast'], dropna=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def feature_group(feature):
    text = str(feature).lower()
    groups = [
        ('intercept', INTERCEPT_KEYWORDS),
        ('missingness_unknown', MISSINGNESS_KEYWORDS),
        ('interaction', INTERACTION_KEYWORDS),
        ('target_inr', TARGET_INR_KEYWORDS),
        ('height_weight', HEIGHT_WEIGHT_KEYWORDS),
        ('genotype', GENOTYPE_KEYWORDS),
        ('race', RACE_KEYWORDS),
        ('gender', GENDER_KEYWORDS),
        ('age', AGE_KEYWORDS),
        ('indication', INDICATION_KEYWORDS),
        ('medication_text_or_flag', MEDICATION_KEYWORDS),
        ('comorbidity_text_or_flag', COMORBIDITY_KEYWORDS),
    ]
    for group, keywords in groups:
        if any(keyword in text for keyword in keywords):
            return group
    return 'other'


def clinical_signal_table(table):
    if table is None or len(table) == 0 or 'feature_group' not in table.columns:
        return pd.DataFrame()
    excluded = {'missingness_unknown', 'intercept'}
    return table[~table['feature_group'].isin(excluded)].reset_index(drop=True)


def feature_group_summary(coef_long):
    group_cols = ['algorithm', 'feature_set', 'reward_scheme', 'coefficient_mode', 'arm_id', 'arm_name', 'feature_group']
    return (
        coef_long.groupby(group_cols, dropna=False)
        .agg(
            n_features=('feature', 'nunique'),
            mean_abs_coefficient=('abs_coefficient', 'mean'),
            max_abs_coefficient=('abs_coefficient', 'max'),
            l1_mass=('abs_coefficient', 'sum'),
        )
        .reset_index()
        .sort_values(['algorithm', 'feature_set', 'arm_id', 'l1_mass'], ascending=[True, True, True, False])
    )


def hybrid_shared_coefficients(
    bandit,
    feature_cols,
    algorithm,
    feature_set,
    reward_scheme='binary',
):
    if not hasattr(bandit, 'beta_shared'):
        return pd.DataFrame()
    beta = np.asarray(bandit.beta_shared, dtype=float)
    rows = []
    for idx, feature in enumerate(feature_cols):
        value = float(beta[idx])
        rows.append({
            'algorithm': algorithm,
            'feature_set': feature_set,
            'reward_scheme': str(reward_scheme),
            'feature_index': int(idx),
            'feature': feature,
            'shared_coefficient': value,
            'abs_shared_coefficient': abs(value),
            'feature_group': feature_group(feature),
        })
    return pd.DataFrame(rows).sort_values('abs_shared_coefficient', ascending=False)


def audit_bandit_coefficients(
    bandit_factory,
    df,
    feature_sets,
    feature_set_name,
    algorithm,
    seed=0,
    reward_scheme='binary',
    standardize=True,
    coefficient_mode='effective',
    top_n=30,
):
    X, y, feature_cols, scale_stats = make_feature_matrix(
        df, feature_sets, feature_set_name, standardize=standardize
    )
    factory_kwargs = {'n_arms': int(len(np.unique(y))), 'n_features': int(X.shape[1])}
    factory_signature = inspect.signature(bandit_factory)
    if 'feature_cols' in factory_signature.parameters:
        factory_kwargs['feature_cols'] = feature_cols
    bandit = bandit_factory(**factory_kwargs)
    bandit, run = fit_bandit_for_audit(
        bandit=bandit,
        X=X,
        y=y,
        seed=seed,
        reward_scheme=reward_scheme,
        name=algorithm,
    )
    coef_long = coefficients_long_table(
        bandit=bandit,
        feature_cols=feature_cols,
        algorithm=algorithm,
        feature_set=feature_set_name,
        reward_scheme=reward_scheme,
        coefficient_mode=coefficient_mode,
    )
    top = top_coefficients(coef_long, top_n=top_n)
    contrasts = coefficient_contrasts(coef_long, top_n=top_n)
    groups = feature_group_summary(coef_long)
    shared = hybrid_shared_coefficients(
        bandit=bandit,
        feature_cols=feature_cols,
        algorithm=algorithm,
        feature_set=feature_set_name,
        reward_scheme=reward_scheme,
    )
    final_accuracy = float(run['is_correct'].mean()) if len(run) else np.nan
    final_reward = float(run['reward'].mean()) if len(run) else np.nan
    audit_summary = pd.DataFrame([{
        'algorithm': algorithm,
        'feature_set': feature_set_name,
        'reward_scheme': str(reward_scheme),
        'seed': int(seed),
        'n_features': int(len(feature_cols)),
        'final_accuracy': final_accuracy,
        'final_mean_reward': final_reward,
    }])
    return {
        'summary': audit_summary,
        'coefficients': coef_long,
        'top_coefficients': top,
        'top_clinical_coefficients': clinical_signal_table(top),
        'contrasts': contrasts,
        'clinical_contrasts': clinical_signal_table(contrasts),
        'feature_group_summary': groups,
        'hybrid_shared_coefficients': shared,
        'hybrid_shared_clinical_coefficients': clinical_signal_table(shared),
        'scale_stats': scale_stats,
        'run': run,
        'bandit': bandit,
        'feature_cols': feature_cols,
    }


def save_coefficient_audit(output_dir, prefix, audit, save_full_coefficients=True, save_run=False):
    tables = {
        f'{prefix}_summary': audit.get('summary'),
        f'{prefix}_top_coefficients': audit.get('top_coefficients'),
        f'{prefix}_top_clinical_coefficients': audit.get('top_clinical_coefficients'),
        f'{prefix}_contrasts': audit.get('contrasts'),
        f'{prefix}_clinical_contrasts': audit.get('clinical_contrasts'),
        f'{prefix}_feature_group_summary': audit.get('feature_group_summary'),
        f'{prefix}_hybrid_shared_coefficients': audit.get('hybrid_shared_coefficients'),
        f'{prefix}_hybrid_shared_clinical_coefficients': audit.get('hybrid_shared_clinical_coefficients'),
        f'{prefix}_scale_stats': audit.get('scale_stats'),
    }
    if save_full_coefficients:
        tables[f'{prefix}_coefficients'] = audit.get('coefficients')
    if save_run:
        tables[f'{prefix}_run'] = audit.get('run')
    return save_benchmark_outputs(output_dir, save_step_results=save_run, **tables)


def coefficient_model_card(coef_long, zero_tolerance=1e-12, top_n=10):
    """Build arm-level sparsity, ranking, and support tables from saved coefficients."""
    required = {
        'algorithm',
        'feature_set',
        'reward_scheme',
        'coefficient_mode',
        'arm_id',
        'arm_name',
        'feature_index',
        'feature',
        'coefficient',
        'feature_group',
    }
    missing = required.difference(coef_long.columns)
    if missing:
        raise ValueError(f'Coefficient table is missing required columns: {sorted(missing)}')
    if zero_tolerance < 0:
        raise ValueError('zero_tolerance must be non-negative.')
    if top_n < 1:
        raise ValueError('top_n must be at least 1.')

    frame = coef_long.copy()
    frame['coefficient'] = pd.to_numeric(frame['coefficient'], errors='raise')
    frame['abs_coefficient'] = frame['coefficient'].abs()
    frame['is_nonzero'] = frame['abs_coefficient'] > float(zero_tolerance)
    frame['is_positive'] = frame['coefficient'] > float(zero_tolerance)
    frame['is_negative'] = frame['coefficient'] < -float(zero_tolerance)

    group_cols = [
        'algorithm',
        'feature_set',
        'reward_scheme',
        'coefficient_mode',
        'arm_id',
        'arm_name',
    ]
    summary = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            n_features=('feature', 'size'),
            n_nonzero=('is_nonzero', 'sum'),
            n_positive=('is_positive', 'sum'),
            n_negative=('is_negative', 'sum'),
            l1_norm=('abs_coefficient', 'sum'),
            l2_norm=('coefficient', lambda values: float(np.linalg.norm(values.to_numpy(dtype=float)))),
            max_abs_coefficient=('abs_coefficient', 'max'),
        )
        .reset_index()
    )
    summary['n_zero'] = summary['n_features'] - summary['n_nonzero']
    summary['nonzero_fraction'] = summary['n_nonzero'] / summary['n_features']
    summary['zero_tolerance'] = float(zero_tolerance)

    strongest_positive = (
        frame[frame['is_positive']]
        .sort_values(group_cols + ['coefficient'], ascending=[True] * len(group_cols) + [False])
        .groupby(group_cols, dropna=False)
        .head(1)[group_cols + ['feature', 'coefficient']]
        .rename(columns={
            'feature': 'strongest_positive_feature',
            'coefficient': 'strongest_positive_coefficient',
        })
    )
    strongest_negative = (
        frame[frame['is_negative']]
        .sort_values(group_cols + ['coefficient'], ascending=[True] * len(group_cols) + [True])
        .groupby(group_cols, dropna=False)
        .head(1)[group_cols + ['feature', 'coefficient']]
        .rename(columns={
            'feature': 'strongest_negative_feature',
            'coefficient': 'strongest_negative_coefficient',
        })
    )
    summary = (
        summary.merge(strongest_positive, how='left', on=group_cols)
        .merge(strongest_negative, how='left', on=group_cols)
        .sort_values(['algorithm', 'feature_set', 'arm_id'])
        .reset_index(drop=True)
    )

    ranked_tables = []
    for direction, mask, ascending in [
        ('positive', frame['is_positive'], False),
        ('negative', frame['is_negative'], True),
    ]:
        ranked = frame[mask].sort_values(
            group_cols + ['coefficient'],
            ascending=[True] * len(group_cols) + [ascending],
        )
        ranked = ranked.groupby(group_cols, dropna=False).head(top_n).copy()
        ranked['direction'] = direction
        ranked['rank'] = ranked.groupby(group_cols, dropna=False).cumcount() + 1
        ranked_tables.append(ranked)

    top_weights = pd.concat(ranked_tables, ignore_index=True)
    top_weights = top_weights[
        group_cols
        + ['direction', 'rank', 'feature_index', 'feature', 'feature_group', 'coefficient', 'abs_coefficient']
    ].sort_values(['algorithm', 'feature_set', 'arm_id', 'direction', 'rank'])

    index_cols = [
        'algorithm',
        'feature_set',
        'reward_scheme',
        'coefficient_mode',
        'feature_index',
        'feature',
        'feature_group',
    ]
    wide = frame.pivot_table(
        index=index_cols,
        columns='arm_name',
        values='coefficient',
        aggfunc='first',
    ).reset_index()
    wide.columns.name = None

    arm_names = [
        arm_name
        for _, arm_name in (
            frame[['arm_id', 'arm_name']]
            .drop_duplicates()
            .sort_values('arm_id')
            .itertuples(index=False, name=None)
        )
    ]
    coefficient_cols = []
    nonzero_cols = []
    for arm_name in arm_names:
        coefficient_col = f'{arm_name}_coefficient'
        nonzero_col = f'{arm_name}_nonzero'
        wide = wide.rename(columns={arm_name: coefficient_col})
        wide[nonzero_col] = wide[coefficient_col].abs() > float(zero_tolerance)
        coefficient_cols.append(coefficient_col)
        nonzero_cols.append(nonzero_col)

    wide['n_nonzero_arms'] = wide[nonzero_cols].sum(axis=1)
    wide['nonzero_arm_pattern'] = wide.apply(
        lambda row: '+'.join(
            arm_name
            for arm_name, nonzero_col in zip(arm_names, nonzero_cols)
            if bool(row[nonzero_col])
        ) or 'none',
        axis=1,
    )
    wide['max_abs_coefficient'] = wide[coefficient_cols].abs().max(axis=1)
    wide['zero_tolerance'] = float(zero_tolerance)
    wide = wide[
        index_cols
        + coefficient_cols
        + nonzero_cols
        + ['n_nonzero_arms', 'nonzero_arm_pattern', 'max_abs_coefficient', 'zero_tolerance']
    ]
    wide = wide.sort_values(
        ['max_abs_coefficient', 'feature_index'],
        ascending=[False, True],
    ).reset_index(drop=True)

    support_patterns = (
        wide.groupby(['nonzero_arm_pattern', 'n_nonzero_arms'], dropna=False)
        .agg(n_features=('feature', 'size'))
        .reset_index()
        .sort_values(['n_nonzero_arms', 'n_features', 'nonzero_arm_pattern'], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    support_patterns['zero_tolerance'] = float(zero_tolerance)

    return {
        'summary': summary,
        'top_weights': top_weights.reset_index(drop=True),
        'coefficients_wide': wide,
        'support_patterns': support_patterns,
    }


def hybrid_shared_model_card(shared_coefficients, zero_tolerance=1e-12, top_n=10):
    """Build the same model card for a HybridRidgeLinUCB shared coefficient vector."""
    required = {
        'algorithm',
        'feature_set',
        'reward_scheme',
        'feature_index',
        'feature',
        'shared_coefficient',
        'feature_group',
    }
    missing = required.difference(shared_coefficients.columns)
    if missing:
        raise ValueError(f'Hybrid shared coefficient table is missing required columns: {sorted(missing)}')

    normalized = shared_coefficients.copy().rename(columns={'shared_coefficient': 'coefficient'})
    normalized['coefficient_mode'] = 'shared_ordinal'
    normalized['arm_id'] = -1
    normalized['arm_name'] = 'shared_ordinal'
    return coefficient_model_card(
        normalized,
        zero_tolerance=zero_tolerance,
        top_n=top_n,
    )


def save_coefficient_model_card(output_dir, prefix, model_card):
    """Save model-card tables without overwriting benchmark output metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for table_name in ['summary', 'top_weights', 'coefficients_wide', 'support_patterns']:
        table = model_card.get(table_name)
        if table is None:
            continue
        path = output_dir / f'{prefix}_model_card_{table_name}.csv'
        table.to_csv(path, index=False)
        paths[table_name] = path
    return paths


def plot_top_coefficients(top_table, title='Top model coefficients', top_n=20):
    if top_table is None or len(top_table) == 0:
        raise ValueError('top_table is empty.')
    plot_df = top_table.sort_values('abs_coefficient', ascending=False).head(top_n).copy()
    labels = plot_df['arm_name'] + ' | ' + plot_df['feature']
    fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(plot_df))))
    ax.barh(np.arange(len(plot_df)), plot_df['coefficient'])
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, linewidth=1)
    ax.set_xlabel('Coefficient value')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.25)
    plt.tight_layout()
    return fig, ax
