import pandas as pd

# For the purpose of this demonstration, I'll generate a sample CSV file with random data.
# You will need to replace 'sample_data.csv' with the path to your actual CSV file.

# Path to the CSV file
csv_file_path = './data/warfarin_no_nulls.csv'

# Read data from the CSV file
df = pd.read_csv(csv_file_path)

# Now, for each column, print the value counts in a verbose manner
for column in df.columns:
    print(f"Value counts for the column '{column}':")
    print(df[column].value_counts())
    print("\n")  # Add a newline for better readability between columns

