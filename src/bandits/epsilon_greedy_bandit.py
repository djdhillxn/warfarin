import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('./../preprocessing/data/warfarin_one_hot_encoded_full_power.csv')

# Define the epsilon-greedy multi-armed bandit
class EpsilonGreedyBandit:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms, dtype=int)  # how many times each arm was pulled
        self.values = np.zeros(n_arms)  # average reward for each arm
        
    def select_arm(self):
        if np.random.rand() > self.epsilon:
            return np.argmax(self.values)
        else:
            return np.random.randint(self.n_arms)
        
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # New estimate is a running average
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward

# Set up the simulation
n_iterations = 20
cumulative_regrets = []
average_accuracies = []

# Get the number of arms from the number of unique labels
n_arms = df['Therapeutic Dose of Warfarin'].nunique()

for i in range(n_iterations):
    # Shuffle the dataset
    df_shuffled = shuffle(df, random_state=i)
    
    # Split the data into features and labels
    X = df_shuffled.drop('Therapeutic Dose of Warfarin', axis=1).values
    y = df_shuffled['Therapeutic Dose of Warfarin'].values
    
    # Initialize the bandit
    bandit = EpsilonGreedyBandit(n_arms)
    
    # Track performance
    correct_predictions = 0
    cumulative_regret = 0
    
    # Simulate the bandit selecting arms and receiving reward
    for j in range(len(df_shuffled)):
        # The bandit makes a prediction (selects an arm)
        chosen_arm = bandit.select_arm()
        
        # Get the actual label
        actual_label = y[j]
        
        # Check if the bandit's prediction was correct
        if chosen_arm == actual_label:
            reward = 1
            correct_predictions += 1
        else:
            reward = 0
            
        # Update the bandit with the reward for the chosen arm
        bandit.update(chosen_arm, reward)
        
        # Calculate regret (difference between the optimal and chosen action)
        optimal_reward = 1  # assuming the optimal action would always be correct
        regret = optimal_reward - reward
        cumulative_regret += regret
    
    # Store the cumulative regret for this iteration
    cumulative_regrets.append(cumulative_regret)
    
    # Calculate accuracy for this iteration
    accuracy = correct_predictions / len(df_shuffled)
    average_accuracies.append(accuracy)
    
    print(f"Iteration {i+1}: Accuracy = {accuracy}, Cumulative Regret = {cumulative_regret}")

# Calculate the average accuracy over all iterations
final_average_accuracy = np.mean(average_accuracies)
print(f"Average Accuracy over {n_iterations} iterations: {final_average_accuracy}")

