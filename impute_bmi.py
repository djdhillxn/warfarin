import numpy as np
import pandas as pd


def dataframe_info(df):
    # General Information
    print("General Information:")
    print("--------------------")
    df.info()
    print("\n")
    '''
    # Descriptive statistics for numerical columns
    print("Descriptive Statistics:")
    print("-----------------------")
    print(df.describe())
    print("\n")
    # Value counts for non-numeric columns
    non_numeric = df.select_dtypes(exclude=[np.number])
    for col in non_numeric.columns:
        print(f"Value Counts for {col}:")
        print("-----------------------")
        print(df[col].value_counts())
        print("\n")
    '''

def dashed_line():
    print('\n')
    print('--------------------------------------------------------------------------------------')
    print('\n')

def sorted_info(df):
    non_null_counts = df.count().sort_values(ascending=False)
    print(f"<class 'pandas.core.frame.DataFrame'>")
    print(f"Shape: {df.shape}")
    print("Columns:")
    for col, count in non_null_counts.items():
        dtype = df[col].dtype
        print(f"{col}: {count} non-null {dtype}")

# Example usage:
# sorted_info(df)


# Assuming `df` is your DataFrame
#dataframe_info(df)
data_path = './data/cleaned_warfarin.csv'
#metadata_path = "/content/drive/MyDrive/warfarin/metadata.xls"
# Reload the CSV file into a DataFrame
df = pd.read_csv(data_path)

dashed_line()

print('df.shape before preprocessing:',df.shape)

dashed_line()


from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

missing_values_before_imputation = df[['Height (cm)', 'Weight (kg)']].isnull().sum()

print("missing_values_before_imputation:")
print(missing_values_before_imputation)

dashed_line()

# Select columns for imputation
data_to_impute = df[['Height (cm)', 'Weight (kg)']]

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_to_impute)

# Impute missing values using KNN
imputer = KNNImputer(n_neighbors=5)
imputed_data = imputer.fit_transform(scaled_data)

# Inverse transform to original scale
imputed_data_original_scale = scaler.inverse_transform(imputed_data)

# Update the original dataframe with imputed values
df['Height (cm)'] = imputed_data_original_scale[:, 0]
df['Weight (kg)'] = imputed_data_original_scale[:, 1]

# Check if there are any missing values left after imputation
missing_values_after_imputation = df[['Height (cm)', 'Weight (kg)']].isnull().sum()

print("missing_values_after_imputation:")
print(missing_values_after_imputation)

dashed_line()

print('df.shape after preprocessing:',df.shape)

dashed_line()

df.to_csv('./data/warfarinx.csv')
print('saved imputed dataset warfarinx.csv')

#dataframe_info(df)

dashed_line()







