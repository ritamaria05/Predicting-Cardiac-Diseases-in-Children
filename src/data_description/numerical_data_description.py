import pandas as pd
import numpy as np


df = pd.read_excel('UCMF(5).xls')

weight = df['Peso']
print("Weight Column Description:")
print(f"Column defined type: {weight.dtype}")
print(f"Total number of lines in the column: {weight.size}")
print(f"Number of non-null values in the column: {weight.count()}")
print(f"Number of null values in the column: {weight.isna().sum()}")
print(f"Percentage of null values in the column: {weight.isna().mean() * 100:.2f}%")

if pd.api.types.is_numeric_dtype(weight):
    print("The column is numeric.")
    print(f"Minimum value in the column: {weight.min()}")
    print(f"Maximum value in the column: {weight.max()}")
    print(f"Mean value in the column: {weight.mean():.2f}")
    print(f"Standard deviation in the column: {weight.std():.2f}")
    print(f"Median value in the column: {weight.median():.2f}")
    print(f"25th percentile value in the column: {weight.quantile(0.25):.2f}")
    print(f"75th percentile value in the column: {weight.quantile(0.75):.2f}")
    print(f"Interquartile range (IQR) in the column: {weight.quantile(0.75) - weight.quantile(0.25):.2f}")
    print(f"Skewness in the column: {weight.skew():.2f}")
    print(f"Zeros in the column: {(weight == 0).sum()}")
    print(f"Negative values in the column: {(weight < 0).sum()}")
else:
    print("The column is not numeric.")
    

height = df['Altura']
print("\n Height Column Description:")
print(f"Column defined type: {height.dtype}")
print(f"Total number of lines in the column: {height.size}")
print(f"Number of non-null values in the column: {height.count()}")
print(f"Number of null values in the column: {height.isna().sum()}")
print(f"Percentage of null values in the column: {height.isna().mean() * 100:.2f}%")

if pd.api.types.is_numeric_dtype(height):
    print("The column is numeric.")
    print(f"Minimum value in the column: {height.min()}")
    print(f"Maximum value in the column: {height.max()}")
    print(f"Mean value in the column: {height.mean():.2f}")
    print(f"Standard deviation in the column: {height.std():.2f}")
    print(f"Median value in the column: {height.median():.2f}")
    print(f"25th percentile value in the column: {height.quantile(0.25):.2f}")
    print(f"75th percentile value in the column: {height.quantile(0.75):.2f}")
    print(f"Interquartile range (IQR) in the column: {height.quantile(0.75) - height.quantile(0.25):.2f}")
    print(f"Skewness in the column: {height.skew():.2f}")
    print(f"Zeros in the column: {(height == 0).sum()}")
    print(f"Negative values in the column: {(height < 0).sum()}")
else:
    print("The column is not numeric.")
    

imc = df['IMC']
print("\n IMC Column Description:")
print(f"Column defined type: {imc.dtype}")
print(f"Total number of lines in the column: {imc.size}")
print(f"Number of non-null values in the column: {imc.count()}")
print(f"Number of null values in the column: {imc.isna().sum()}")
print(f"Percentage of null values in the column: {imc.isna().mean() * 100:.2f}%")

if pd.api.types.is_numeric_dtype(imc):
    print("The column is numeric.")
    print(f"Minimum value in the column: {imc.min()}")
    print(f"Maximum value in the column: {imc.max()}")
    print(f"Mean value in the column: {imc.mean():.2f}")
    print(f"Standard deviation in the column: {imc.std():.2f}")
    print(f"Median value in the column: {imc.median():.2f}")
    print(f"25th percentile value in the column: {imc.quantile(0.25):.2f}")
    print(f"75th percentile value in the column: {imc.quantile(0.75):.2f}")
    print(f"Interquartile range (IQR) in the column: {imc.quantile(0.75) - imc.quantile(0.25):.2f}")
    print(f"Skewness in the column: {imc.skew():.2f}")
    print(f"Zeros in the column: {(imc == 0).sum()}")
    print(f"Negative values in the column: {(imc < 0).sum()}")
    print(f"Infinite values in the column: {imc.isin([float('inf'), float('-inf')]).sum()}")
    print("\n[Alert for potential data issues]")
    print(f"IMC < 10:   {(imc < 10).sum()}")
    print(f"IMC > 70:   {(imc > 70).sum()}")
else:
    print("The column is not numeric.")
if {'Peso', 'Altura'}.issubset(df.columns):
    mask = (df['Peso'] > 0) & (df['Altura'] > 0) & (df['IMC'] > 0) & np.isfinite(df['IMC']) # this line creates a boolean mask to filter rows where 'Peso', 'Altura', and 'IMC' are all positive and 'IMC' is finite
    imc_calculated = df.loc[mask, 'Peso'] / ((df.loc[mask, 'Altura']/100) ** 2)
    differences = (np.abs(df.loc[mask, 'IMC'] - imc_calculated) > 0.5).sum()
    print(f"\nDifferences > 0.5 kg/m²: {differences}") # this line will print the number of rows where the difference between the provided IMC and the calculated IMC is greater than 0.5 kg/m²
    
age = df['IDADE']
print("\n Age Column Description:")
print(f"Column defined type: {age.dtype}")
print(f"Total number of lines in the column: {age.size}")
print(f"Number of non-null values in the column: {age.count()}")
print(f"Number of null values in the column: {age.isna().sum()}")
print(f"Percentage of null values in the column: {age.isna().mean() * 100:.2f}%")

if pd.api.types.is_numeric_dtype(age):
    print("The column is numeric.")
    print(f"Minimum value in the column: {age.min()}")
    print(f"Maximum value in the column: {age.max()}")
    print(f"Mean value in the column: {age.mean():.2f}")
    print(f"Standard deviation in the column: {age.std():.2f}")
    print(f"Median value in the column: {age.median():.2f}")
    print(f"25th percentile value in the column: {age.quantile(0.25):.2f}")
    print(f"75th percentile value in the column: {age.quantile(0.75):.2f}")
    print(f"Interquartile range (IQR) in the column: {age.quantile(0.75) - age.quantile(0.25):.2f}")
    print(f"Skewness in the column: {age.skew():.2f}")
    print(f"Zeros in the column: {(age == 0).sum()}")
    print(f"Negative values in the column: {(age < 0).sum()}")
    print(f"Infinite values in the column: {age.isin([float('inf'), float('-inf')]).sum()}")
else:
    print("The column is not numeric.")
    print("Non-numeric values in the column:")
    print(df.loc[pd.to_numeric(df['IDADE'], errors='coerce').isna() & df['IDADE'].notna(), 'IDADE'].value_counts())
    age_numeric = pd.to_numeric(df['IDADE'], errors='coerce')
    print(f"Minimum value in the column: {age_numeric.min()}")
    print(f"Maximum value in the column: {age_numeric.max()}")
    print(f"Mean value in the column: {age_numeric.mean():.2f}")
    print(f"Standard deviation in the column: {age_numeric.std():.2f}")
    print(f"Median value in the column: {age_numeric.median():.2f}")
    print(f"25th percentile value in the column: {age_numeric.quantile(0.25):.2f}")
    print(f"75th percentile value in the column: {age_numeric.quantile(0.75):.2f}")
    print(f"Interquartile range (IQR) in the column: {age_numeric.quantile(0.75) - age_numeric.quantile(0.25):.2f}")
    print(f"Skewness in the column: {age_numeric.skew():.2f}")
    print(f"Zeros in the column: {(age_numeric == 0).sum()}")
    print(f"Negative values in the column: {(age_numeric < 0).sum()}")
    print(f"Infinite values in the column: {age_numeric.isin([float('inf'), float('-inf')]).sum()}")
    
    
pa_sistolic = df['PA SISTOLICA']
print("\n PA Sistolic Column Description:")
print(f"Column defined type: {pa_sistolic.dtype}")
print(f"Total number of lines in the column: {pa_sistolic.size}")
print(f"Number of non-null values in the column: {pa_sistolic.count()}")
print(f"Number of null values in the column: {pa_sistolic.isna().sum()}")
print(f"Percentage of null values in the column: {pa_sistolic.isna().mean() * 100:.2f}%")

if pd.api.types.is_numeric_dtype(pa_sistolic):
    print("The column is numeric.")
    print(f"Minimum value in the column: {pa_sistolic.min()}")
    print(f"Maximum value in the column: {pa_sistolic.max()}")
    print(f"Mean value in the column: {pa_sistolic.mean():.2f}")
    print(f"Standard deviation in the column: {pa_sistolic.std():.2f}")
    print(f"Median value in the column: {pa_sistolic.median():.2f}")
    print(f"25th percentile value in the column: {pa_sistolic.quantile(0.25):.2f}")
    print(f"75th percentile value in the column: {pa_sistolic.quantile(0.75):.2f}")
    print(f"Interquartile range (IQR) in the column: {pa_sistolic.quantile(0.75) - pa_sistolic.quantile(0.25):.2f}")
    print(f"Skewness in the column: {pa_sistolic.skew():.2f}")
    print(f"Zeros in the column: {(pa_sistolic == 0).sum()}")
    print(f"Negative values in the column: {(pa_sistolic < 0).sum()}")
else:
    print("The column is not numeric.")
    
pa_diastolic = df['PA DIASTOLICA']
print("\n PA Diastolic Column Description:")
print(f"Column defined type: {pa_diastolic.dtype}")
print(f"Total number of lines in the column: {pa_diastolic.size}")
print(f"Number of non-null values in the column: {pa_diastolic.count()}")
print(f"Number of null values in the column: {pa_diastolic.isna().sum()}")
print(f"Percentage of null values in the column: {pa_diastolic.isna().mean() * 100:.2f}%")

if pd.api.types.is_numeric_dtype(pa_diastolic):
    print("The column is numeric.")
    print(f"Minimum value in the column: {pa_diastolic.min()}")
    print(f"Maximum value in the column: {pa_diastolic.max()}")
    print(f"Mean value in the column: {pa_diastolic.mean():.2f}")
    print(f"Standard deviation in the column: {pa_diastolic.std():.2f}")
    print(f"Median value in the column: {pa_diastolic.median():.2f}")
    print(f"25th percentile value in the column: {pa_diastolic.quantile(0.25):.2f}")
    print(f"75th percentile value in the column: {pa_diastolic.quantile(0.75):.2f}")
    print(f"Interquartile range (IQR) in the column: {pa_diastolic.quantile(0.75) - pa_diastolic.quantile(0.25):.2f}")
    print(f"Skewness in the column: {pa_diastolic.skew():.2f}")
    print(f"Zeros in the column: {(pa_diastolic == 0).sum()}")
    print(f"Negative values in the column: {(pa_diastolic < 0).sum()}")
else:
    print("The column is not numeric.")
    
fc = df['FC']
print("\n FC Column Description:")
print(f"Column defined type: {fc.dtype}")
print(f"Total number of lines in the column: {fc.size}")
print(f"Number of non-null values in the column: {fc.count()}")
print(f"Number of null values in the column: {fc.isna().sum()}")
print(f"Percentage of null values in the column: {fc.isna().mean() * 100:.2f}%")

if pd.api.types.is_numeric_dtype(fc):
    print("The column is numeric.")
    print(f"Minimum value in the column: {fc.min()}")
    print(f"Maximum value in the column: {fc.max()}")
    print(f"Mean value in the column: {fc.mean():.2f}")
    print(f"Standard deviation in the column: {fc.std():.2f}")
    print(f"Median value in the column: {fc.median():.2f}")
    print(f"25th percentile value in the column: {fc.quantile(0.25):.2f}")
    print(f"75th percentile value in the column: {fc.quantile(0.75):.2f}")
    print(f"Interquartile range (IQR) in the column: {fc.quantile(0.75) - fc.quantile(0.25):.2f}")
    print(f"Skewness in the column: {fc.skew():.2f}")
    print(f"Zeros in the column: {(fc == 0).sum()}")
    print(f"Negative values in the column: {(fc < 0).sum()}")
else:
    print("The column is not numeric.")
    print("Non-numeric values in the column:")
    print(df.loc[pd.to_numeric(df['FC'], errors='coerce').isna() & df['FC'].notna(), 'FC'].value_counts())
    fc_numeric = pd.to_numeric(df['FC'], errors='coerce')
    print(f"Minimum value in the column: {fc_numeric.min()}")
    print(f"Maximum value in the column: {fc_numeric.max()}")
    print(f"Mean value in the column: {fc_numeric.mean():.2f}")
    print(f"Standard deviation in the column: {fc_numeric.std():.2f}")
    print(f"Median value in the column: {fc_numeric.median():.2f}")
    print(f"25th percentile value in the column: {fc_numeric.quantile(0.25):.2f}")
    print(f"75th percentile value in the column: {fc_numeric.quantile(0.75):.2f}")
    print(f"Interquartile range (IQR) in the column: {fc_numeric.quantile(0.75) - fc_numeric.quantile(0.25):.2f}")
    print(f"Skewness in the column: {fc_numeric.skew():.2f}")
    print(f"Zeros in the column: {(fc_numeric == 0).sum()}")
    print(f"Negative values in the column: {(fc_numeric < 0).sum()}")
    print(f"Infinite values in the column: {fc_numeric.isin([float('inf'), float('-inf')]).sum()}")