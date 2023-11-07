import pandas as pd
import sys

def print_value_counts(df):
    for column in df.columns:
        print(f"Value counts for column: {column}")
        print("-" * 40)
        # Include null values in the counts
        counts = df[column].value_counts(dropna=False)
        print(counts)
        print("-" * 40)

def main(csv_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Print the value counts for each column
    print_value_counts(df)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py path_to_csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    main(csv_path)

