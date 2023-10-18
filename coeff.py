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


data_path = './data/warfarinx.csv'
#metadata_path = "/content/drive/MyDrive/warfarin/metadata.xls"
# Reload the CSV file into a DataFrame
df = pd.read_csv(data_path)

dashed_line()

print('df.shape before preprocessing:',df.shape)

dashed_line()


bins = pd.IntervalIndex.from_tuples([(0, 20.9999), (20.9999, 49), (49, 20000)])
df["Therapeutic Dose of Warfarin"] = pd.cut(df["Therapeutic Dose of Warfarin"], bins)

df['VKORC1_coeff'] = df['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'].apply(
    lambda x: -1.6974 if x == 'A/A' else
              -0.8677 if x == 'A/G' else
              -0.4854 if x == 'Unknown' else 0
)

df['Race_coefficient_clinical'] = df['Race'].apply(
    lambda x: -0.6752 if x == 'Asian' else
               0.4060 if x == 'Black or African American' else
               0.0443 if x == 'Unknown' else 0
)
df['Race_coefficient_pharma'] = df['Race'].apply(
    lambda x: -1.0920 if x == 'Asian' else
              -0.2760 if x == 'Black or African American' else
              -0.1032 if x == 'Unknown' else 0
)
df['Cyp2C9 genotypes_coefficient'] = df['Cyp2C9 genotypes'].apply(
    lambda x: -0.5211 if x == '*1/*2' else
              -0.9357 if x == '*1/*3' else
              -1.0616 if x == '*2/*2' else
              -1.9206 if x == '*2/*3' else
              -2.3312 if x == '*3/*3' else
              -0.2188 if x == 'Unknown' else 0  # Assuming coefficient is 0 for '*1/*1' and other unspecified genotypes
)


df['Clinical Dose'] = (   4.0376
                        - 0.2546*df['Age']
                        + 0.0118*df['Height (cm)']
                        + 0.0134*df['Weight (kg)']
                        + 1.0000*df['Race_coefficient_clinical']
                        + 1.2799*df['Enzyme inducer status']
                        - 0.5695*df['Amiodarone (Cordarone)']
                      )
df['Clinical Dose'] =  df['Clinical Dose']*df['Clinical Dose']
df['Clinical Dose'] = pd.cut(df['Clinical Dose'], bins)


df['Pharmacogenetic Dose'] = (    5.6044
                                - 0.2614*df['Age']
                                + 0.0087*df['Height (cm)']
                                + 0.0128*df['Weight (kg)']
                                + 1.0000*df['Race_coefficient_pharma']
                                + 1.1816*df['Enzyme inducer status']
                                - 0.5503*df['Amiodarone (Cordarone)']
                                + 1.0000*df['Cyp2C9 genotypes_coefficient']
                                + 1.0000*df['VKORC1_coeff']
)
df['Pharmacogenetic Dose'] =  df['Pharmacogenetic Dose']*df['Pharmacogenetic Dose']
df['Pharmacogenetic Dose'] = pd.cut(df['Pharmacogenetic Dose'], bins)


df['Fixed Dose'] =  35
df['Fixed Dose'] = pd.cut(df['Fixed Dose'], bins)


df['Fixed Correct'] = df['Fixed Dose'] == df["Therapeutic Dose of Warfarin"]
print("df['Fixed Correct'].value_counts()")
df['Fixed Correct'].value_counts()

dashed_line()

df['Clinical Correct'] = df['Clinical Dose'] == df["Therapeutic Dose of Warfarin"]
print("df['Clinical Correct'].value_counts()")
df['Clinical Correct'].value_counts()

dashed_line()

df['Pharmacogenetic Correct'] = df['Pharmacogenetic Dose'] == df["Therapeutic Dose of Warfarin"]
print("df['Pharmacogenetic Correct'].value_counts()")
df['Pharmacogenetic Correct'].value_counts()


dashed_line()
