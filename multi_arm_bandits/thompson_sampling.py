import numpy as np
import pandas as pd
from sklearn.utils import shuffle

class ThompsonSamplingBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.successes = np.ones(n_arms)  # Success counts for each arm
        self.failures = np.ones(n_arms)  # Failure counts for each arm
        
    def select_arm(self):
        # Sample from the posterior for each arm
        theta_sample = np.random.beta(self.successes, self.failures)
        return np.argmax(theta_sample)
    
    def update(self, chosen_arm, reward):
        if reward == 1:
            self.successes[chosen_arm] += 1
        else:
            self.failures[chosen_arm] += 1

# Load the dataset
df = pd.read_csv('./../preprocessing/data/warfarin_one_hot_encoded_full_power.csv')

# Define parameters
n_iterations = 20
cumulative_regrets = []
average_accuracies = []

# Get the number of arms from the number of unique labels
n_arms = df['Therapeutic Dose of Warfarin'].nunique()

for i in range(n_iterations):
    df_shuffled = shuffle(df, random_state=i)
    X = df_shuffled.drop('Therapeutic Dose of Warfarin', axis=1).values
    y = df_shuffled['Therapeutic Dose of Warfarin'].values
    
    # Initialize the Thompson Sampling bandit
    ts_bandit = ThompsonSamplingBandit(n_arms)
    
    correct_predictions = 0
    cumulative_regret = 0
    
    # Simulate the decision process
    for j in range(len(df_shuffled)):
        chosen_arm = ts_bandit.select_arm()
        actual_label = y[j]
        
        reward = 1 if chosen_arm == actual_label else 0
        correct_predictions += reward
        
        ts_bandit.update(chosen_arm, reward)
        
        optimal_reward = 1  # assuming the optimal action would always be correct
        regret = optimal_reward - reward
        cumulative_regret += regret
    
    cumulative_regrets.append(cumulative_regret)
    
    accuracy = correct_predictions / len(df_shuffled)
    average_accuracies.append(accuracy)
    
    print(f"Iteration {i+1}: Accuracy = {accuracy}, Cumulative Regret = {cumulative_regret}")

final_average_accuracy = np.mean(average_accuracies)
print(f"Average Accuracy over {n_iterations} iterations: {final_average_accuracy}")

