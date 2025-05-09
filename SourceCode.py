import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from google.colab import files
uploaded = files.upload()

import pandas as pd

try:
    df = pd.read_csv('AirQualityUCI.csv')
    display(df.head())
except FileNotFoundError:
    print("Error: 'AirQualityUCI.csv' not found.")
    df = None
except pd.errors.ParserError:
    print("Error: Could not parse the CSV file. Check file format.")
    df = None
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    df = None

import pandas as pd

try:
    df = pd.read_csv('AirQualityUCI.csv', sep=';')
    display(df.head())
except FileNotFoundError:
    print("Error: 'AirQualityUCI.csv' not found.")
    df = None
except pd.errors.ParserError:
    print("Error: Could not parse the CSV file. Check file format.")
    df = None
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    df = None

# Examine the shape of the DataFrame
print("Shape of the DataFrame:", df.shape)

# Check data types and identify potential inconsistencies
print("\nData types of each column:")
print(df.info())

# Summarize numerical features
print("\nSummary statistics for numerical columns:")
print(df.describe())

# Initial exploration of non-numerical features
print("\nValue counts for 'Date' column:")
print(df['Date'].value_counts())
print("\nValue counts for 'Time' column:")
print(df['Time'].value_counts())

# Check for missing values
print("\nNumber of missing values in each column:")
print(df.isnull().sum())


# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 15', 'Unnamed: 16'])

# Replace -200 with NaN
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].replace(-200, np.nan)

# Impute missing values using median for numeric columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Convert columns to numeric types, handling errors
for col in ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH', 'NMHC(GT)','NOx(GT)','NO2(GT)']:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

# Impute any remaining NaN values with the median for those columns
for col in ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH', 'NMHC(GT)','NOx(GT)','NO2(GT)']:
    df[col] = df[col].fillna(df[col].median())

# Outlier handling using IQR method (example for 'CO(GT)')
Q1 = df['CO(GT)'].quantile(0.25)
Q3 = df['CO(GT)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['CO(GT)'] = np.clip(df['CO(GT)'], lower_bound, upper_bound)

display(df.head())
display(df.info())

# Calculate descriptive statistics
numerical_cols = df.select_dtypes(include=['number']).columns
descriptive_stats = df[numerical_cols].describe()
print("Descriptive Statistics:\n", descriptive_stats)

# Calculate the correlation matrix
correlation_matrix = df[numerical_cols].corr()
print("\nCorrelation Matrix:\n", correlation_matrix)

# Analyze 'Date' and 'Time' columns
print("\nAnalysis of Date and Time columns:")
print("Missing Values in 'Date':", df['Date'].isnull().sum())
print("Missing Values in 'Time':", df['Time'].isnull().sum())
print("Unique Dates:", df['Date'].nunique())
print("Unique Times:", df['Time'].nunique())

# Attempt to convert 'Date' and 'Time' to datetime
try:
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S', errors='coerce')
    print("Successfully converted 'Date' and 'Time' to datetime.")
except ValueError as e:
    print(f"Error converting 'Date' and 'Time' to datetime: {e}")

# Analyze air quality indicators
air_quality_indicators = ['CO(GT)', 'NOx(GT)', 'NO2(GT)']
for indicator in air_quality_indicators:
    print(f"\nAnalysis for {indicator}:")
    print(df[indicator].describe())
    if 'DateTime' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
            df.set_index('Date', inplace=True)
            print(df[indicator].resample('D').mean())
        except Exception as e:
            print(f"Error performing time-based analysis for {indicator}: {e}")
    else:
        print("Datetime conversion failed. Time-based analysis skipped.")

# ✅ Imports


# After uploading, load the CSV file into a DataFrame
df = pd.read_csv('AirQualityUCI.csv', sep=';')  # Replace with the name of the uploaded file

# ✅ Clean column names (if necessary)
df.columns = df.columns.str.strip()

# ✅ Replace commas with periods in all numeric columns and convert them to numeric
def clean_and_convert_column(column):
    # Replace commas with periods in strings (if any)
    column = column.replace({',': '.'}, regex=True)
    # Convert to numeric and coerce errors to NaN
    return pd.to_numeric(column, errors='coerce')

# Clean and convert all relevant columns (CO(GT), and the feature columns)
df['CO(GT)'] = clean_and_convert_column(df['CO(GT)'])
df['C6H6(GT)'] = clean_and_convert_column(df['C6H6(GT)'])
df['T'] = clean_and_convert_column(df['T'])
df['RH'] = clean_and_convert_column(df['RH'])
df['AH'] = clean_and_convert_column(df['AH'])
df['NMHC(GT)'] = clean_and_convert_column(df['NMHC(GT)'])
df['NOx(GT)'] = clean_and_convert_column(df['NOx(GT)'])
df['NO2(GT)'] = clean_and_convert_column(df['NO2(GT)'])

# ✅ Remove rows with NaN values in relevant columns
df = df.dropna(subset=['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH', 'NMHC(GT)', 'NOx(GT)', 'NO2(GT)'])

# ✅ Select features and target
features = ['C6H6(GT)', 'T', 'RH', 'AH', 'NMHC(GT)', 'NOx(GT)', 'NO2(GT)']
target = 'CO(GT)'

X = df[features]
y = df[target]

# ✅ Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

# ✅ Train Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# ✅ Evaluate both models
def evaluate(name, y_true, y_pred):
    print(f"\n🔍 {name} Evaluation:")
    print("R² Score:", r2_score(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))

evaluate("Linear Regression", y_test, lr_preds)
evaluate("Random Forest Regressor", y_test, rf_preds)



# Model names
models = ['Linear Regression', 'Random Forest']

# Mock-up performance scores (replace with your actual values if available)
r2_scores = [0.65, 0.89]      # R² scores
rmse_scores = [1.8, 1.1]      # RMSE values

# Create side-by-side plots for R² and RMSE
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# R² Score Bar Chart
ax[0].bar(models, r2_scores, color=['skyblue', 'seagreen'])
ax[0].set_title('Model Comparison (R² Score)')
ax[0].set_ylim(0, 1)
ax[0].set_ylabel('R² Score')

# RMSE Bar Chart
ax[1].bar(models, rmse_scores, color=['skyblue', 'seagreen'])
ax[1].set_title('Model Comparison (RMSE)')
ax[1].set_ylim(0, 2)
ax[1].set_ylabel('RMSE')

plt.tight_layout()
plt.show()


# Histograms
plt.figure(figsize=(20, 15))
for i, col in enumerate(['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']):
    plt.subplot(4, 4, i + 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')

plt.tight_layout()
plt.show()

# Box Plots
plt.figure(figsize=(20, 15))
for i, col in enumerate(['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']):
    plt.subplot(4, 4, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot of {col}')

plt.tight_layout()
plt.show()


# Scatter Plots
plt.figure(figsize=(15, 10))
pairs = [('CO(GT)', 'PT08.S1(CO)'), ('NMHC(GT)', 'PT08.S2(NMHC)'), ('NOx(GT)', 'PT08.S3(NOx)'), ('NO2(GT)', 'PT08.S4(NO2)'), ('T', 'RH'), ('T', 'AH'), ('RH', 'AH')]
for i, (col1, col2) in enumerate(pairs):
  plt.subplot(2, 4, i + 1)
  sns.scatterplot(x=df[col1], y=df[col2])
  plt.title(f'{col1} vs. {col2}')

plt.tight_layout()
plt.show()

# Time Series Plot (if 'DateTime' column exists)
if 'DateTime' in df.columns:
    plt.figure(figsize=(10, 6))
    df.set_index('DateTime', inplace = True)
    plt.plot(df['CO(GT)'])
    plt.xlabel('DateTime')
    plt.ylabel('CO(GT)')
    plt.title('Time Series of CO(GT)')
    plt.show()
else:
    print("DateTime column not found. Skipping time series plot.")

#!pip install gradio


# Load the dataset
df = pd.read_csv("AirQualityUCI.csv", sep=';')

# Clean data (example: remove unnamed columns and missing values)
df = df.drop(columns=["Unnamed: 15", "Unnamed: 16"], errors='ignore').dropna()

# Sample prediction function (replace with your actual model logic)
def predict(hour):
    try:
        row = df[df['Time'] == hour].iloc[0]
        return f"CO: {row['CO(GT)']}, NOx: {row['NOx(GT)']}, Temp: {row['T']}"
    except IndexError:
        return "No data available for that hour."

# Gradio UI
iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Enter Time (e.g., 18.00.00)"),
    outputs="text",
    title="Air Quality Prediction"
)

iface.launch()
