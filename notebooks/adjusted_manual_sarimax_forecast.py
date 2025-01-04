
# Import Libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error

# Load the dataset
file_path = r"C:\Users\c.hakker\OneDrive - VISTA college\Senior Stuff\Opleiding Data science\Data\merged_tables_new.xlsx"
df = pd.read_excel(file_path)

# Filter for 'C Manufacturing'
branch_name = 'C Manufacturing'
df = df[df['BedrijfstakkenBranchesSBI2008'].str.strip().str.lower() == branch_name.strip().lower()]

# Process 'Year' and 'Quarter' columns
df['Year'] = df['Year'].astype(int)
df['Quarter'] = df['Quarter'].astype(int)
df['Date'] = pd.PeriodIndex.from_fields(year=df['Year'], quarter=df['Quarter'], freq='Q').to_timestamp()

# Set 'Date' as index
df.set_index('Date', inplace=True)
df = df.loc[~df.index.duplicated(keep='first')]

# Define SARIMAX parameters
target_column = '80072ned_Ziekteverzuimpercentage_1'
q1_order = (1, 1, 1)
q1_seasonal_order = (1, 1, 1, 4)
q2_order = (1, 1, 0)
q2_seasonal_order = (0, 1, 1, 4)

# Split data into training and testing
train_end_year = 2021
train_df = df[df['Year'] <= train_end_year]
test_df = df[df['Year'].isin([2022, 2023])]

# Function to perform rolling forecast
def rolling_forecast(train_data, test_data, order, seasonal_order):
    predictions = []
    rolling_train = train_data.copy()
    for date in test_data.index:
        model = sm.tsa.SARIMAX(
            rolling_train[target_column],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)
        forecast = results.get_forecast(steps=1).predicted_mean
        predictions.append(forecast.iloc[0])
        rolling_train = pd.concat([rolling_train, test_data.loc[[date]]])
    return pd.Series(predictions, index=test_data.index)

# Predictions for Q1 2023
y_pred_q1_2023 = rolling_forecast(train_df, test_df[(test_df['Year'] == 2023) & (test_df['Quarter'] == 1)], q1_order, q1_seasonal_order)
mae_q1_2023 = mean_absolute_error(test_df[(test_df['Year'] == 2023) & (test_df['Quarter'] == 1)][target_column], y_pred_q1_2023)
print(f"MAE for Q1 2023: {mae_q1_2023:.4f}")

# Predictions for Q2 2023
y_pred_q2_2023 = rolling_forecast(train_df, test_df[(test_df['Year'] == 2023) & (test_df['Quarter'] == 2)], q2_order, q2_seasonal_order)
mae_q2_2023 = mean_absolute_error(test_df[(test_df['Year'] == 2023) & (test_df['Quarter'] == 2)][target_column], y_pred_q2_2023)
print(f"MAE for Q2 2023: {mae_q2_2023:.4f}")

# Save results
predictions = pd.DataFrame({
    'Date': pd.concat([y_pred_q1_2023.index, y_pred_q2_2023.index]),
    'Predicted': pd.concat([y_pred_q1_2023, y_pred_q2_2023])
})
predictions.to_csv("manual_predictions.csv", index=False)
