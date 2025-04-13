import pandas as pd
import sys

def print_value_counts(df):
    # Calculate the null counts for each column
    null_counts = df.isnull().sum()
    # Sort the columns based on null counts (ascending)
    sorted_columns = null_counts.sort_values().index

    for column in sorted_columns:
        print(f"Value counts for column: {column}")
        print("-" * 40)
        # Include null values in the counts
        counts = df[column].value_counts(dropna=False)
        print(counts)
        print("-" * 40)

def main(csv_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Print the shape of the DataFrame
    print(f"The DataFrame shape is: {df.shape}")
    print("-" * 40)
    
    # Print the value counts for each column
    print_value_counts(df)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py path_to_csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    main(csv_path)

