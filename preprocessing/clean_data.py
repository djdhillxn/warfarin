import numpy as np
import pandas as pd


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

# Example usage:
# sorted_info(df)


# Assuming `df` is your DataFrame
#dataframe_info(df)
data_path = './data/warfarin.csv'
#metadata_path = "/content/drive/MyDrive/warfarin/metadata.xls"
# Reload the CSV file into a DataFrame
df = pd.read_csv(data_path)

dashed_line()

print('df.shape before preprocessing:',df.shape)

dashed_line()

#dataframe_info(df)
print("# Step 1: Drop rows where 'Therapeutic Dose of Warfarin' is null")
print("df.shape:",df.shape)
print("null value counts for warfarin dosage:",df['Therapeutic Dose of Warfarin'].isnull().sum())
df = df[df['Therapeutic Dose of Warfarin'].notna()]
print('null value counts for dose after dropping:', df['Therapeutic Dose of Warfarin'].isnull().sum())
print("df.shape after dropping:",df.shape)


dashed_line()


print("# Step 2: Drop the unnamed columns and PharmGKB Subject ID")
unnamed_columns = ['PharmGKB Subject ID', 'Unnamed: 63', 'Unnamed: 64', 'Unnamed: 65']
print("df.shape:",df.shape)
print('dropping empty columns...')
df.drop(columns=unnamed_columns, inplace=True)
print("df.shape after dropping:",df.shape)


dashed_line()


print("# Step 3: Replace null values in 'Gender'")
print(df['Gender'].value_counts())
print('null values counts for gender:',df['Gender'].isnull().sum())
print('imputing null value counts with mode: ',df['Gender'].mode()[0])
df['Gender'].fillna('male', inplace=True)
print('null value counts for gender after imputation:', df['Gender'].isnull().sum())
print(df['Gender'].value_counts())

dashed_line()

print("# Step 4: Impute missing 'Age' values with the mode")
print(df['Age'].value_counts())
print('null value counts for age:', df['Age'].isnull().sum())
mode_age = df['Age'].mode()[0]
print('imputing the null value counts with mode age value:',mode_age)
df['Age'].fillna(mode_age, inplace=True)
print('null value counts for age after imputation:', df['Age'].isnull().sum())
print(df['Age'].value_counts())

df['Age'] = df['Age'].map({'10 - 19': 1, '20 - 29': 2, '30 - 39': 3, '40 - 49': 4, '50 - 59': 5, '60 - 69': 6, '70 - 79': 7, '80 - 89' : 8, '90+' : 9})




dashed_line()




print('''
# Step 5: Replace all NaN values in 'Amiodarone (Cordarone)' with 0, effectively implementing the assumption
# "When amiodarone use was unknown, it was assumed that it was not used."
''')
print(df['Amiodarone (Cordarone)'].value_counts())
print('null value counts before imputation',df['Amiodarone (Cordarone)'].isnull().sum())
df['Amiodarone (Cordarone)'].fillna(0, inplace=True)
print('null value counts after imputation',df['Amiodarone (Cordarone)'].isnull().sum())
print(df['Amiodarone (Cordarone)'].value_counts())

dashed_line()

print('''
# Step 6: filling Carbamazepine, Phenytoin, Rifampin null values with 0
''')

print("df['Carbamazepine (Tegretol)'].isnull().sum()",df['Carbamazepine (Tegretol)'].isnull().sum())
print("df['Phenytoin (Dilantin)'].isnull().sum()",df['Phenytoin (Dilantin)'].isnull().sum())
print("df['Rifampin or Rifampicin'].isnull().sum()",df['Rifampin or Rifampicin'].isnull().sum())
df['Carbamazepine (Tegretol)'].fillna(0, inplace=True)
df['Phenytoin (Dilantin)'].fillna(0, inplace=True)
df['Rifampin or Rifampicin'].fillna(0, inplace=True)
df['Carbamazepine (Tegretol)'] = df['Carbamazepine (Tegretol)'].map({1.0: True, 0.0: False})
df['Phenytoin (Dilantin)'] = df['Phenytoin (Dilantin)'].map({1.0: True, 0.0:False})
df['Rifampin or Rifampicin'] = df['Rifampin or Rifampicin'].map({1.0: True, 0.0:False})
df['Enzyme inducer status'] = df["Carbamazepine (Tegretol)"] | df["Phenytoin (Dilantin)"] | df["Rifampin or Rifampicin"]
df['Enzyme inducer status'] = df['Enzyme inducer status'].map({True: 1.0, False: 0.0})
print("df['Carbamazepine (Tegretol)'].isnull().sum()",df['Carbamazepine (Tegretol)'].isnull().sum())
print("df['Phenytoin (Dilantin)'].isnull().sum()",df['Phenytoin (Dilantin)'].isnull().sum())
print("df['Rifampin or Rifampicin'].isnull().sum()",df['Rifampin or Rifampicin'].isnull().sum())


dashed_line()


print('''
# Step 7: Filling null values in 'Cyp2C9 genotypes' and 'CYP2C9 consensus' with 'Unknown'
''')
print("df['Cyp2C9 genotypes'].isnull().sum() before imputation",df['Cyp2C9 genotypes'].isnull().sum())
print("df['CYP2C9 consensus'].isnull().sum() before imputation",df['CYP2C9 consensus'].isnull().sum())
df['Cyp2C9 genotypes'].fillna('Unknown', inplace=True)
df['CYP2C9 consensus'].fillna('Unknown', inplace=True)
print("df['Cyp2C9 genotypes'].isnull().sum() after imputation",df['Cyp2C9 genotypes'].isnull().sum())
print("df['CYP2C9 consensus'].isnull().sum() after imputation",df['CYP2C9 consensus'].isnull().sum())


dashed_line()



print('''
# Step 8: Imputing missing values for column "VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T"
''')
print('# Check for the current state of the "VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T" column')
column_name = "VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T"
print("unique_entries before imputing:", df[column_name].unique())
print("num null values before imputing:",df[column_name].isnull().sum())
print("value counts before imputing:",df[column_name].value_counts())



# Correct the column names based on the actual dataset
rs2359612_col = 'VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G'
rs9934438_col = 'VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G'
rs8050894_col = 'VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G'

# Update the conditions
condition_1a = (df['Race'] != "Black or African American") & (df['Race'] != "Missing or Mixed Race") & (df[rs2359612_col] == 'C/C')
condition_1b = (df['Race'] != "Black or African American") & (df['Race'] != "Missing or Mixed Race") & (df[rs2359612_col] == 'T/T')
condition_1c = (df['Race'] != "Black or African American") & (df['Race'] != "Missing or Mixed Race") & (df[rs2359612_col] == 'C/T')

condition_2a = df[rs9934438_col] == 'C/C'
condition_2b = df[rs9934438_col] == 'T/T'
condition_2c = df[rs9934438_col] == 'C/T'

condition_3a = (df['Race'] != "Black or African American") & (df['Race'] != "Missing or Mixed Race") & (df[rs8050894_col] == 'G/G')
condition_3b = (df['Race'] != "Black or African American") & (df['Race'] != "Missing or Mixed Race") & (df[rs8050894_col] == 'C/C')
condition_3c = (df['Race'] != "Black or African American") & (df['Race'] != "Missing or Mixed Race") & (df[rs8050894_col] == 'C/G')

# Apply imputations
df.loc[condition_1a & df[column_name].isna(), column_name] = 'G/G'
df.loc[condition_1b & df[column_name].isna(), column_name] = 'A/A'
df.loc[condition_1c & df[column_name].isna(), column_name] = 'A/G'

df.loc[condition_2a & df[column_name].isna(), column_name] = 'G/G'
df.loc[condition_2b & df[column_name].isna(), column_name] = 'A/A'
df.loc[condition_2c & df[column_name].isna(), column_name] = 'A/G'

df.loc[condition_3a & df[column_name].isna(), column_name] = 'G/G'
df.loc[condition_3b & df[column_name].isna(), column_name] = 'A/A'
df.loc[condition_3c & df[column_name].isna(), column_name] = 'A/G'

print("# Filling null values that can't be imputed in 'VKORC1 genotype' with 'Unknown'")
df['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'].fillna('Unknown', inplace=True)

print("unique_entries after imputing:", df[column_name].unique())
print("value counts after imputing:",df[column_name].value_counts())
print("num null values after imputing:",df[column_name].isnull().sum())



dashed_line()

print('df.shape',df.shape)

dashed_line()

sorted_info(df)

dashed_line()

dataframe_info(df)

dashed_line()

df.to_csv('./data/cleaned_warfarin.csv', index=False)

print('cleaned dataset successfully saved')
