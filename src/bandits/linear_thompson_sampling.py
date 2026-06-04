import numpy as np


class LinearThompsonSampling:
    """
    Disjoint Linear Thompson Sampling for contextual bandits.

    Each arm keeps a Bayesian/ridge-style linear reward model:
        A_a = lambda I + sum x x^T
        b_a = sum r x
        theta_hat_a = A_a^{-1} b_a

    At decision time we sample a plausible theta for each arm and choose the
    arm with the largest sampled payoff:
        theta_tilde_a ~ N(theta_hat_a, sample_scale^2 A_a^{-1})
        a_t = argmax_a x_t^T theta_tilde_a

    covariance_type='diagonal' uses only the diagonal of A_a^{-1}. This is fast
    and usually stable, but it ignores feature correlations.

    covariance_type='full' samples from the full covariance A_a^{-1}. The class
    maintains Cholesky factors of A_a^{-1} and updates them with a rank-one
    downdate after each Sherman-Morrison inverse update, so full covariance
    sampling is much faster than recomputing a matrix factorization at every
    decision.
    """

    def __init__(
        self,
        n_arms,
        n_features,
        lambda_reg=10.0,
        sample_scale=0.05,
        covariance_type='diagonal',
        random_state=None,
        jitter=1e-10,
    ):
        if lambda_reg <= 0:
            raise ValueError('lambda_reg must be positive.')
        if sample_scale < 0:
            raise ValueError('sample_scale must be non-negative.')
        covariance_type = self._normalize_covariance_type(covariance_type)

        self.n_arms = int(n_arms)
        self.n_features = int(n_features)
        self.lambda_reg = float(lambda_reg)
        self.sample_scale = float(sample_scale)
        self.covariance_type = covariance_type
        self.jitter = float(jitter)
        self.rng = np.random.default_rng(random_state)

        eye = np.identity(self.n_features)
        self.A = np.repeat((self.lambda_reg * eye)[None, :, :], self.n_arms, axis=0)
        self.A_inv = np.repeat(((1.0 / self.lambda_reg) * eye)[None, :, :], self.n_arms, axis=0)
        self.b = np.zeros((self.n_arms, self.n_features))
        self.theta = np.zeros((self.n_arms, self.n_features))

        self._chol_A_inv = None
        self._chol_dirty = None
        if self.covariance_type == 'full':
            self._initialize_cholesky_cache()

    @staticmethod
    def _normalize_covariance_type(covariance_type):
        aliases = {
            'diag': 'diagonal',
            'diagonal': 'diagonal',
            'full': 'full',
            'full_covariance': 'full',
        }
        if covariance_type not in aliases:
            raise ValueError("covariance_type must be one of {'diagonal', 'diag', 'full', 'full_covariance'}.")
        return aliases[covariance_type]

    def _initialize_cholesky_cache(self):
        self._chol_A_inv = np.zeros_like(self.A_inv)
        self._chol_dirty = np.ones(self.n_arms, dtype=bool)

    def _safe_cholesky(self, matrix):
        matrix = 0.5 * (matrix + matrix.T)
        eye = np.identity(matrix.shape[0])
        jitter = self.jitter
        for _ in range(6):
            try:
                return np.linalg.cholesky(matrix + jitter * eye)
            except np.linalg.LinAlgError:
                jitter *= 10.0
        return np.linalg.cholesky(matrix + jitter * eye)

    @staticmethod
    def _cholesky_downdate_lower(L, vector, eps=1e-12):
        """
        Return lower-triangular chol(L L^T - vector vector^T).

        This is used because the Sherman-Morrison inverse update is exactly a
        rank-one covariance downdate:
            A_inv_new = A_inv_old - v v^T
        where v = A_inv_old x / sqrt(1 + x^T A_inv_old x).
        """
        L_new = L.copy()
        x = vector.astype(float, copy=True)
        n = L_new.shape[0]

        for k in range(n):
            diag_sq = L_new[k, k] * L_new[k, k] - x[k] * x[k]
            if diag_sq <= eps:
                raise np.linalg.LinAlgError('Cholesky downdate lost positive definiteness.')
            r = np.sqrt(diag_sq)
            c = r / L_new[k, k]
            s = x[k] / L_new[k, k]
            L_new[k, k] = r
            if k + 1 < n:
                L_new[k + 1:, k] = (L_new[k + 1:, k] - s * x[k + 1:]) / c
                x[k + 1:] = c * x[k + 1:] - s * L_new[k + 1:, k]

        return L_new

    def _ensure_cholesky(self, arm):
        if self._chol_A_inv is None:
            self._initialize_cholesky_cache()
        if self._chol_dirty[arm]:
            self._chol_A_inv[arm] = self._safe_cholesky(self.A_inv[arm])
            self._chol_dirty[arm] = False
        return self._chol_A_inv[arm]

    def _sample_theta_diagonal(self):
        posterior_std = np.sqrt(np.maximum(np.diagonal(self.A_inv, axis1=1, axis2=2), 0.0))
        noise = self.rng.normal(size=self.theta.shape)
        return self.theta + self.sample_scale * posterior_std * noise

    def _sample_theta_full(self):
        samples = np.empty_like(self.theta)
        for arm in range(self.n_arms):
            L = self._ensure_cholesky(arm)
            noise = self.rng.normal(size=self.n_features)
            samples[arm] = self.theta[arm] + self.sample_scale * (L @ noise)
        return samples

    def select_arm(self, x):
        x = np.asarray(x, dtype=float)
        if self.sample_scale == 0:
            sampled_theta = self.theta
        elif self.covariance_type == 'full':
            sampled_theta = self._sample_theta_full()
        else:
            sampled_theta = self._sample_theta_diagonal()
        scores = sampled_theta @ x
        return int(np.argmax(scores))

    def update(self, chosen_arm, x, reward):
        x = np.asarray(x, dtype=float)
        reward = float(reward)
        arm = int(chosen_arm)

        A_inv_x = self.A_inv[arm] @ x
        denominator = 1.0 + float(x.T @ A_inv_x)
        downdate_vector = A_inv_x / np.sqrt(max(denominator, 1e-12))

        self.A_inv[arm] -= np.outer(A_inv_x, A_inv_x) / denominator
        self.A_inv[arm] = 0.5 * (self.A_inv[arm] + self.A_inv[arm].T)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        self.theta[arm] = self.A_inv[arm] @ self.b[arm]

        if self.covariance_type == 'full' and self._chol_A_inv is not None and not self._chol_dirty[arm]:
            try:
                self._chol_A_inv[arm] = self._cholesky_downdate_lower(
                    self._chol_A_inv[arm], downdate_vector
                )
            except np.linalg.LinAlgError:
                self._chol_dirty[arm] = True


class LinearTS(LinearThompsonSampling):
    """Short alias used in notebooks."""

    pass
