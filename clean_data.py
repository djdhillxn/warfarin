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


print("# Step 2: Drop the unnamed columns")
unnamed_columns = ['Unnamed: 63', 'Unnamed: 64', 'Unnamed: 65']
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

print('df.shape',df.shape)

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

dataframe_info(df)

dashed_line()
