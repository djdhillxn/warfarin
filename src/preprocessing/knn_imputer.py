import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import sys

# Function to add a visual separator in output
def dashed_line():
    print('\n')
    print('-'*69)
    print('\n')

# Function to print dataframe information
def dataframe_info(df):
    print("General Information:")
    print("--------------------")
    df.info()
    print("\n")

# Read the data from the provided path
data_path = sys.argv[1]  # Using command line argument for file path
df = pd.read_csv(data_path)

dashed_line()

# Print the dataframe's shape before preprocessing
print('df.shape before preprocessing:', df.shape)

dashed_line()

# Print missing values before imputation
missing_values_before_imputation = df[['Height (cm)', 'Weight (kg)']].isnull().sum()
print("missing_values_before_imputation:")
print(missing_values_before_imputation)

dashed_line()

# Identify categorical columns, you will need to update this list based on your dataset
categorical_columns = df.select_dtypes(include=['object', 'category']).columns

# Apply one-hot encoding to categorical columns
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_encoded)

# Impute missing values using KNN
imputer = KNNImputer(n_neighbors=5)
imputed_data = imputer.fit_transform(scaled_data)

# Inverse transform to original scale
imputed_data_original_scale = scaler.inverse_transform(imputed_data)

# Update the original dataframe with imputed values
# Make sure that the indexes correspond to the right columns after encoding
df['Height (cm)'] = imputed_data_original_scale[:, df_encoded.columns.get_loc('Height (cm)')]
df['Weight (kg)'] = imputed_data_original_scale[:, df_encoded.columns.get_loc('Weight (kg)')]

# Check if there are any missing values left after imputation
missing_values_after_imputation = df[['Height (cm)', 'Weight (kg)']].isnull().sum()
print("missing_values_after_imputation:")
print(missing_values_after_imputation)

dashed_line()

# Print the dataframe's shape after preprocessing
print('df.shape after preprocessing:', df.shape)

dashed_line()

# Save the dataframe to a new CSV file
output_path = './data/warfarinx.csv'
df.to_csv(output_path, index=False)
print(f'saved imputed dataset to {output_path}')

dashed_line()

