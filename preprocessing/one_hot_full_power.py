import pandas as pd

# For the purpose of this demonstration, I'll generate a sample CSV file with random data.
# You will need to replace 'sample_data.csv' with the path to your actual CSV file.


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


# Path to the CSV file
csv_file_path = './data/warfarinx.csv'

# Read data from the CSV file
df = pd.read_csv(csv_file_path)

# Define the bin edges directly
bins = [0, 20.9999, 49, 20000] 
# Labels for the bins as integers
labels = [0, 1, 2]
# Cut the 'Therapeutic Dose of Warfarin' column into bins with the integer labels
df["Therapeutic Dose of Warfarin"] = pd.cut(df["Therapeutic Dose of Warfarin"], bins=bins, labels=labels, include_lowest=True)

df.drop(columns=['Comorbidities','Medications'], inplace=True)

"""
columns_to_drop = ['Comorbidities', 'Medications', 'INR on Reported Therapeutic Dose of Warfarin', 'Target INR', 'Indication for Warfarin Treatment', 'Estimated Target INR Range Based on Indication']
print("df.shape before dropping:",df.shape)
print("dropping: ['Comorbidities', 'Medications', 'INR on Reported Therapeutic Dose of Warfarin', 'Target INR', 'Indication for Warfarin Treatment', 'Estimated Target INR Range Based on Indication']")
df.drop(columns=columns_to_drop, inplace=True)
print("df.shape after dropping:",df.shape)
"""

dashed_line()

print('df: warfarin_full_power.csv')
print('df.shape before preprocessing:',df.shape)

dashed_line()

# Now, for each column, print the value counts in a verbose manner
for column in df.columns:
    print(f"Value counts for the column '{column}':")
    print(df[column].value_counts())
    print("\n")  # Add a newline for better readability between columns

# Let's assume 'Label' is the name of your labels column
# And 'Height' and 'Weight' are your continuous variables
categorical_columns = df.columns.difference(['Height (cm)', 'Weight (kg)', 'Therapeutic Dose of Warfarin'])

# Apply one-hot encoding to your categorical columns
df_encoded = pd.get_dummies(df, columns=categorical_columns, dummy_na=True)

# Save the one-hot encoded DataFrame
df_encoded.to_csv('./data/warfarin_one_hot_encoded_full_power.csv', index=False)

dashed_line()

# Now, for each column, print the value counts in a verbose manner
for column in df_encoded.columns:
    print(f"Value counts for the column '{column}':")
    print(df_encoded[column].value_counts())
    print("\n")  # Add a newline for better readability between columns

dashed_line()

print('new df: warfarin_one_hot_encoded_full_power.csv')
print('new_df.shape after getting:',df_encoded.shape)

dashed_line()
