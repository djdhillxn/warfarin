import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
import warnings


class LassoUCBBandit:
    """
    Practical LASSO + UCB contextual bandit for sparse/high-dimensional features.

    This class is intentionally designed for the warfarin simulation setting:
    rewards are scalar online feedback values, each arm has its own sparse linear
    model, and the LASSO-selected active set is used to compute a compact ridge
    uncertainty bonus.

    Decision rule after the warmup/forced-sampling phase:
        score_a(x) = x^T theta_lasso_a + alpha * sqrt(x_S^T A_S^{-1} x_S)

    where S is the active feature set selected by LASSO for arm a. The intercept
    column can be excluded from L1 shrinkage by passing intercept_index.
    """

    def __init__(
        self,
        n_arms,
        n_features,
        lasso_alpha=0.001,
        ucb_alpha=0.0,
        lambda_reg=10.0,
        min_samples_per_arm=20,
        refit_frequency=25,
        max_iter=2500,
        tol=1e-4,
        coefficient_tol=1e-8,
        intercept_index=None,
        random_state=None,
    ):
        if lasso_alpha <= 0:
            raise ValueError('lasso_alpha must be positive.')
        if lambda_reg <= 0:
            raise ValueError('lambda_reg must be positive.')
        if refit_frequency <= 0:
            raise ValueError('refit_frequency must be positive.')

        self.n_arms = int(n_arms)
        self.n_features = int(n_features)
        self.lasso_alpha = float(lasso_alpha)
        self.ucb_alpha = float(ucb_alpha)
        self.lambda_reg = float(lambda_reg)
        self.min_samples_per_arm = int(min_samples_per_arm)
        self.refit_frequency = int(refit_frequency)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.coefficient_tol = float(coefficient_tol)
        self.intercept_index = None if intercept_index is None else int(intercept_index)
        self.rng = np.random.default_rng(random_state)

        self.X_hist = [[] for _ in range(self.n_arms)]
        self.r_hist = [[] for _ in range(self.n_arms)]
        self.arm_counts = np.zeros(self.n_arms, dtype=int)
        self.total_updates = 0

        self.theta = np.zeros((self.n_arms, self.n_features))
        self.intercept_terms = np.zeros(self.n_arms)
        self.selected_features = [self._initial_selected_features() for _ in range(self.n_arms)]
        self.active_A_inv = [self._initial_active_inverse(indices) for indices in self.selected_features]
        self.last_fit_update = np.full(self.n_arms, -1, dtype=int)
        self.models = [None for _ in range(self.n_arms)]

    def _initial_selected_features(self):
        if self.intercept_index is not None:
            return np.array([self.intercept_index], dtype=int)
        return np.arange(self.n_features, dtype=int)

    def _initial_active_inverse(self, selected):
        n_active = max(1, len(selected))
        return (1.0 / self.lambda_reg) * np.identity(n_active)

    def _fit_columns(self):
        if self.intercept_index is None:
            return np.arange(self.n_features, dtype=int)
        return np.array([i for i in range(self.n_features) if i != self.intercept_index], dtype=int)

    def _needs_forced_sample(self):
        return np.any(self.arm_counts < self.min_samples_per_arm)

    def _forced_sample_arm(self):
        min_count = self.arm_counts.min()
        candidates = np.flatnonzero(self.arm_counts == min_count)
        return int(candidates[self.total_updates % len(candidates)])

    def _make_model(self, arm):
        if self.models[arm] is None:
            self.models[arm] = Lasso(
                alpha=self.lasso_alpha,
                fit_intercept=self.intercept_index is not None,
                max_iter=self.max_iter,
                tol=self.tol,
                selection='cyclic',
                warm_start=True,
                random_state=None,
            )
        return self.models[arm]

    def _refresh_arm_model(self, arm):
        if self.arm_counts[arm] == 0:
            return

        X_arm = np.asarray(self.X_hist[arm], dtype=float)
        r_arm = np.asarray(self.r_hist[arm], dtype=float)
        fit_cols = self._fit_columns()
        X_fit = X_arm[:, fit_cols]
        model = self._make_model(arm)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            model.fit(X_fit, r_arm)

        theta = np.zeros(self.n_features)
        theta[fit_cols] = model.coef_
        if self.intercept_index is not None:
            theta[self.intercept_index] = model.intercept_
            self.intercept_terms[arm] = 0.0
        else:
            self.intercept_terms[arm] = model.intercept_

        active = fit_cols[np.abs(model.coef_) > self.coefficient_tol]
        if self.intercept_index is not None:
            active = np.unique(np.r_[self.intercept_index, active]).astype(int)
        elif len(active) == 0:
            active = np.arange(self.n_features, dtype=int)

        X_active = X_arm[:, active]
        A_active = self.lambda_reg * np.identity(len(active)) + X_active.T @ X_active
        try:
            A_inv = np.linalg.inv(A_active)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A_active)

        self.theta[arm] = theta
        self.selected_features[arm] = active
        self.active_A_inv[arm] = A_inv
        self.last_fit_update[arm] = self.total_updates

    def _refresh_due_models(self):
        for arm in range(self.n_arms):
            enough = self.arm_counts[arm] >= max(2, self.min_samples_per_arm)
            due = (self.total_updates - self.last_fit_update[arm]) >= self.refit_frequency
            never_fit = self.last_fit_update[arm] < 0
            if enough and (due or never_fit):
                self._refresh_arm_model(arm)

    def select_arm(self, x):
        if self._needs_forced_sample():
            return self._forced_sample_arm()

        self._refresh_due_models()
        x = np.asarray(x, dtype=float)
        means = self.theta @ x + self.intercept_terms

        if self.ucb_alpha == 0:
            return int(np.argmax(means))

        bonuses = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            active = self.selected_features[arm]
            x_active = x[active]
            bonuses[arm] = np.sqrt(max(float(x_active.T @ self.active_A_inv[arm] @ x_active), 0.0))
        return int(np.argmax(means + self.ucb_alpha * bonuses))

    def update(self, chosen_arm, x, reward):
        arm = int(chosen_arm)
        self.X_hist[arm].append(np.asarray(x, dtype=float).copy())
        self.r_hist[arm].append(float(reward))
        self.arm_counts[arm] += 1
        self.total_updates += 1

        if self.arm_counts[arm] >= max(2, self.min_samples_per_arm):
            due = (self.total_updates - self.last_fit_update[arm]) >= self.refit_frequency
            if due or self.last_fit_update[arm] < 0:
                self._refresh_arm_model(arm)

    def active_feature_counts(self):
        return [int(len(features)) for features in self.selected_features]


class LassoLinUCB(LassoUCBBandit):
    """Named alias for readability in notebooks."""

    pass
