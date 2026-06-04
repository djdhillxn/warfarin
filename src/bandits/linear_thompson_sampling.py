import numpy as np


class LinearThompsonSampling:
    """
    Disjoint Linear Thompson Sampling for contextual bandits.

    Each arm keeps a Bayesian/ridge-style linear reward model:
        A_a = lambda I + sum x x^T
        b_a = sum r x
        theta_hat_a = A_a^{-1} b_a

    At decision time we sample a plausible theta for each arm and choose the
    arm with the largest sampled payoff. By default we use a diagonal posterior
    approximation for speed on repeated warfarin simulations; set
    covariance_type='full' for the full Gaussian covariance.
    """

    def __init__(
        self,
        n_arms,
        n_features,
        lambda_reg=10.0,
        sample_scale=0.05,
        covariance_type='diagonal',
        random_state=None,
    ):
        if lambda_reg <= 0:
            raise ValueError('lambda_reg must be positive.')
        if sample_scale < 0:
            raise ValueError('sample_scale must be non-negative.')
        if covariance_type not in {'diagonal', 'full'}:
            raise ValueError("covariance_type must be either 'diagonal' or 'full'.")

        self.n_arms = int(n_arms)
        self.n_features = int(n_features)
        self.lambda_reg = float(lambda_reg)
        self.sample_scale = float(sample_scale)
        self.covariance_type = covariance_type
        self.rng = np.random.default_rng(random_state)

        eye = np.identity(self.n_features)
        self.A = np.repeat((self.lambda_reg * eye)[None, :, :], self.n_arms, axis=0)
        self.A_inv = np.repeat(((1.0 / self.lambda_reg) * eye)[None, :, :], self.n_arms, axis=0)
        self.b = np.zeros((self.n_arms, self.n_features))
        self.theta = np.zeros((self.n_arms, self.n_features))

    def _sample_theta_diagonal(self):
        posterior_std = np.sqrt(np.maximum(np.diagonal(self.A_inv, axis1=1, axis2=2), 0.0))
        noise = self.rng.normal(size=self.theta.shape)
        return self.theta + self.sample_scale * posterior_std * noise

    def _sample_theta_full(self):
        samples = np.empty_like(self.theta)
        jitter = 1e-10 * np.identity(self.n_features)
        for arm in range(self.n_arms):
            cov = (self.sample_scale ** 2) * self.A_inv[arm]
            try:
                samples[arm] = self.rng.multivariate_normal(self.theta[arm], cov, check_valid='ignore')
            except np.linalg.LinAlgError:
                samples[arm] = self.rng.multivariate_normal(self.theta[arm], cov + jitter, check_valid='ignore')
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
        self.A_inv[arm] -= np.outer(A_inv_x, A_inv_x) / denominator
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        self.theta[arm] = self.A_inv[arm] @ self.b[arm]


class LinearTS(LinearThompsonSampling):
    """Short alias used in notebooks."""

    pass
