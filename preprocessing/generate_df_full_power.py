import pandas as pd
import numpy as np

def dataframe_info(df):
    # General Information
    print("General Information:")
    print("--------------------")
    df.info()
    print("\n")
    
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
        print("null value counts:",df[col].isnull().sum())
        print("\n")

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


# Replace 'your_data.csv' with the path to your actual CSV file
csv_file_path = './data/warfarinx.csv'
output_csv_file_path = './data/warfarin_full_power.csv'

# Read data from the CSV file
df = pd.read_csv(csv_file_path)

dashed_line()
 
print('df.shape before preprocessing:',df.shape)

dashed_line()


# Specify the column names you want to drop in a list
columns_to_drop = ['PharmGKB Subject ID',]

# Drop the columns in place
df.drop(columns=columns_to_drop, inplace=True)

# Now, df will have the specified columns dropped, and this operation will not return anything.


# Define the bin edges directly
bins = [0, 20.9999, 49, 20000]

# Labels for the bins as integers
labels = [0, 1, 2]

# Cut the 'Therapeutic Dose of Warfarin' column into bins with the integer labels
df["Therapeutic Dose of Warfarin"] = pd.cut(df["Therapeutic Dose of Warfarin"], bins=bins, labels=labels, include_lowest=True)



#bins = pd.IntervalIndex.from_tuples([(0, 20.9999), (20.9999, 49), (49, 20000)])
#df["Therapeutic Dose of Warfarin"] = pd.cut(df["Therapeutic Dose of Warfarin"], bins)

# Create a new DataFrame with only the columns that have no null values
df_no_nulls = df.loc[:, df.notnull().all()]

# Save the new DataFrame to a new CSV file
df_no_nulls.to_csv(output_csv_file_path, index=False)



dashed_line()

print('df_no_nulls.shape after preprocessing:',df_no_nulls.shape)
 
dashed_line()

sorted_info(df_no_nulls)

dashed_line()

print(f'The new DataFrame without null columns has been saved to {output_csv_file_path}')

dashed_line()
