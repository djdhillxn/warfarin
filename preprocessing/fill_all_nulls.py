import pandas as pd

# Load the CSV file
df = pd.read_csv('./data/warfarinx.csv')


# Fill all NaN values with 'Unknown'
df_filled = df.fillna('Unknown')

# Save the modified DataFrame to a new CSV file
df_filled.to_csv('./data/warfarin_final.csv', index=False)
