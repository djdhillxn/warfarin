import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm


class LinUCB:
    """
    Disjoint LinUCB for K discrete arms.

    The public API is intentionally kept backward compatible with the original
    project code: LinUCB(alpha, n_arms, n_features), select_arm(x), update(...).

    Implementation note:
    The old version explicitly formed inv(A) at every decision. This version
    keeps A_inv updated with the Sherman-Morrison rank-one update, avoiding
    repeated matrix inversion while preserving the same LinUCB scores up to
    floating-point precision when lambda_reg=1.0.
    """
    #def __init__(self, alpha, n_arms, n_features):
    def __init__(self, alpha, n_arms, n_features, lambda_reg=1.0):
        if lambda_reg <= 0:
            raise ValueError('lambda_reg must be positive so A is invertible.')

        self.alpha = alpha
        self.n_arms = n_arms
        self.n_features = n_features
        #self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.lambda_reg = lambda_reg
        self.A = np.repeat((lambda_reg * np.identity(n_features))[None, :, :], n_arms, axis=0)
        self.A_inv = np.repeat(((1.0 / lambda_reg) * np.identity(n_features))[None, :, :], n_arms, axis=0)
        self.b = np.zeros((n_arms, n_features))
        self.theta = np.zeros((n_arms, n_features))

    def select_arm(self, x):
        x = np.asarray(x, dtype=float)
        A_inv_x = self.A_inv @ x
        mean_reward = self.theta @ x
        uncertainty = np.sqrt(np.maximum(np.einsum('ij,j->i', A_inv_x, x), 0.0))
        p = mean_reward + self.alpha * uncertainty
        return int(np.argmax(p))

    def update(self, chosen_arm, x, reward):
        #x = np.array(x, dtype=float)  # Ensure x is of type float
        #reward = float(reward)  # Ensure reward is a float
        #self.A[chosen_arm] += np.outer(x, x)  # Use np.outer to get the outer product
        x = np.asarray(x, dtype=float)
        reward = float(reward)

        A_inv_x = self.A_inv[chosen_arm] @ x
        denominator = 1.0 + float(x.T @ A_inv_x)
        self.A_inv[chosen_arm] -= np.outer(A_inv_x, A_inv_x) / denominator

        self.A[chosen_arm] += np.outer(x, x)
        self.b[chosen_arm] += reward * x
        self.theta[chosen_arm] = self.A_inv[chosen_arm] @ self.b[chosen_arm]

    """
    def update(self, chosen_arm, x, reward):
        self.A[chosen_arm] += x @ x.T
        self.b[chosen_arm] += reward * x
    """

class RidgeLinUCB(LinUCB):
    """
    Named alias for LinUCB when we explicitly tune lambda_reg as ridge strength.
    """

    pass


if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv('./../preprocessing/data/warfarin_one_hot_encoded_full_power.csv')

    # Define parameters
    alpha = 1.0  # Exploration parameter
    n_iterations = 20
    cumulative_regrets = []
    average_accuracies = []

    # Get the number of arms and features
    n_arms = df['Therapeutic Dose of Warfarin'].nunique()
    n_features = df.shape[1] - 1  # Number of features is total columns minus the label column

    for i in range(n_iterations):
        df_shuffled = shuffle(df, random_state=i)
        X = df_shuffled.drop('Therapeutic Dose of Warfarin', axis=1).values
        y = df_shuffled['Therapeutic Dose of Warfarin'].values

        # Initialize the LinUCB model
        linucb = LinUCB(alpha, n_arms, n_features)

        correct_predictions = 0
        cumulative_regret = 0

        for j in tqdm(range(len(df_shuffled)), desc=f'Iteration {i+1}'):
            x = X[j]
            chosen_arm = linucb.select_arm(x)
            actual_label = y[j]

            reward = 1 if chosen_arm == actual_label else 0
            correct_predictions += reward

            linucb.update(chosen_arm, x, reward)

            optimal_reward = 1  # assuming the optimal action would always be correct
            regret = optimal_reward - reward
            cumulative_regret += regret

        cumulative_regrets.append(cumulative_regret)

        accuracy = correct_predictions / len(df_shuffled)
        average_accuracies.append(accuracy)

        print(f"Iteration {i+1}: Accuracy = {accuracy}, Cumulative Regret = {cumulative_regret}")

    final_average_accuracy = np.mean(average_accuracies)
    print(f"Average Accuracy over {n_iterations} iterations: {final_average_accuracy}")
