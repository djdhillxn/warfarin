import numpy as np


class HybridRidgeLinUCB:
    """
    Hybrid ridge LinUCB with arm-specific coefficients plus a shared ordinal dose component.

    The disjoint RidgeLinUCB model learns one coefficient vector per arm:
        score_a(x) = x^T theta_a

    This hybrid variant keeps that arm-specific part and adds one shared ordinal
    direction learned from low/high updates:
        score_a(x) = x^T theta_a + shared_weight * level_a * x^T beta

    where level_a defaults to [-1, 0, 1] for low, medium, high. The shared beta
    lets low and high dose arms borrow signal through an ordered dose direction,
    while the disjoint theta_a vectors preserve arm-specific behavior.
    """

    def __init__(
        self,
        alpha,
        n_arms,
        n_features,
        lambda_arm=10.0,
        lambda_shared=10.0,
        alpha_shared=None,
        shared_weight=1.0,
        arm_levels=None,
    ):
        if lambda_arm <= 0:
            raise ValueError('lambda_arm must be positive.')
        if lambda_shared <= 0:
            raise ValueError('lambda_shared must be positive.')
        if shared_weight < 0:
            raise ValueError('shared_weight must be non-negative.')

        self.alpha = float(alpha)
        self.alpha_shared = float(alpha if alpha_shared is None else alpha_shared)
        self.n_arms = int(n_arms)
        self.n_features = int(n_features)
        self.lambda_arm = float(lambda_arm)
        self.lambda_shared = float(lambda_shared)
        self.shared_weight = float(shared_weight)

        if arm_levels is None:
            if self.n_arms == 3:
                arm_levels = [-1.0, 0.0, 1.0]
            else:
                arm_levels = np.linspace(-1.0, 1.0, self.n_arms)
        self.arm_levels = np.asarray(arm_levels, dtype=float)
        if len(self.arm_levels) != self.n_arms:
            raise ValueError('arm_levels must have length n_arms.')

        eye = np.identity(self.n_features)
        self.A_arm = np.repeat((self.lambda_arm * eye)[None, :, :], self.n_arms, axis=0)
        self.A_arm_inv = np.repeat(((1.0 / self.lambda_arm) * eye)[None, :, :], self.n_arms, axis=0)
        self.b_arm = np.zeros((self.n_arms, self.n_features))
        self.theta = np.zeros((self.n_arms, self.n_features))

        self.A_shared = self.lambda_shared * eye.copy()
        self.A_shared_inv = (1.0 / self.lambda_shared) * eye.copy()
        self.b_shared = np.zeros(self.n_features)
        self.beta_shared = np.zeros(self.n_features)

    def select_arm(self, x):
        x = np.asarray(x, dtype=float)

        arm_means = self.theta @ x
        shared_projection = float(self.beta_shared @ x)
        means = arm_means + self.shared_weight * self.arm_levels * shared_projection

        A_inv_x = self.A_arm_inv @ x
        arm_uncertainty = np.sqrt(np.maximum(np.einsum('ij,j->i', A_inv_x, x), 0.0))

        shared_uncertainty = np.sqrt(max(float(x.T @ self.A_shared_inv @ x), 0.0))
        shared_bonus = self.alpha_shared * np.abs(self.arm_levels) * shared_uncertainty

        scores = means + self.alpha * arm_uncertainty + self.shared_weight * shared_bonus
        return int(np.argmax(scores))

    def update(self, chosen_arm, x, reward):
        x = np.asarray(x, dtype=float)
        reward = float(reward)
        arm = int(chosen_arm)

        # Arm-specific ridge update.
        A_inv_x = self.A_arm_inv[arm] @ x
        denominator = 1.0 + float(x.T @ A_inv_x)
        self.A_arm_inv[arm] -= np.outer(A_inv_x, A_inv_x) / denominator
        self.A_arm[arm] += np.outer(x, x)
        self.b_arm[arm] += reward * x
        self.theta[arm] = self.A_arm_inv[arm] @ self.b_arm[arm]

        # Shared ordinal update. Medium arm has level 0 by default, so it does
        # not distort the low-vs-high shared direction.
        level = float(self.arm_levels[arm])
        if level != 0.0 and self.shared_weight != 0.0:
            z = level * x
            A_inv_z = self.A_shared_inv @ z
            denominator = 1.0 + float(z.T @ A_inv_z)
            self.A_shared_inv -= np.outer(A_inv_z, A_inv_z) / denominator
            self.A_shared += np.outer(z, z)
            self.b_shared += reward * z
            self.beta_shared = self.A_shared_inv @ self.b_shared

    def effective_theta(self):
        """Return the per-arm coefficient matrix actually used in mean scores."""
        return self.theta + self.shared_weight * self.arm_levels[:, None] * self.beta_shared[None, :]


class HybridLinUCB(HybridRidgeLinUCB):
    """Short alias for notebook readability."""

    pass
