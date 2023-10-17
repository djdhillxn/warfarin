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


dataframe_info(df)

dashed_line()
