import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class LinUCB:
    def __init__(self, alpha, n_arms, n_features):
        self.alpha = alpha
        self.n_arms = n_arms
        self.n_features = n_features
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]
    
    def select_arm(self, x):
        p = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            p[arm] = theta.T @ x + self.alpha * np.sqrt(x.T @ A_inv @ x)
        return np.argmax(p)
    

    def update(self, chosen_arm, x, reward):
        x = np.array(x, dtype=float)  # Ensure x is of type float
        reward = float(reward)  # Ensure reward is a float
        self.A[chosen_arm] += np.outer(x, x)  # Use np.outer to get the outer product
        self.b[chosen_arm] += reward * x
    
    """
    def update(self, chosen_arm, x, reward):
        self.A[chosen_arm] += x @ x.T
        self.b[chosen_arm] += reward * x
    """

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

