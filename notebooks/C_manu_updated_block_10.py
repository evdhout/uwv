
# %%
# 0. Import Libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error
import joblib

# %%
# 1. Load the updated dataset
file_path = r"C:\Users\c.hakker\OneDrive - VISTA college\Senior Stuff\Opleiding Data science\Data\merged_tables_new.xlsx"
df = pd.read_excel(file_path)

# %%
# 2. Filter for the Specific Branch 'C Manufacturing'
branch_name = 'C Manufacturing'
df = df[df['BedrijfstakkenBranchesSBI2008'].str.strip().str.lower() == branch_name.strip().lower()]

# %%
# 3. Verify and Process the 'Year' and 'Quarter' Columns
df['Year'] = df['Year'].astype(int)
df['Quarter'] = df['Quarter'].astype(int)
df['Date'] = pd.PeriodIndex.from_fields(year=df['Year'], quarter=df['Quarter'], freq='Q').to_timestamp()

# %%
# 4. Set 'Date' as Index
df.set_index('Date', inplace=True, drop=True)
df = df.loc[~df.index.duplicated(keep='first')]
df.index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='QS')

# %%
# 5. Define Parameters for SARIMAX
target_column = '80072ned_Ziekteverzuimpercentage_1'

# %%
# 6. Split the Data into Training, Validation, and Testing Sets
train_end_year = 2021
train_df = df[df['Year'] <= train_end_year].copy()
val_df = df[df['Year'] == 2022].copy()
test_df = df[df['Year'].isin([2023, 2024])].copy()
y_train = train_df[target_column]
y_val = val_df[target_column]
y_test = test_df[target_column]

# %%
# 7. Rolling Forecast Function
def rolling_forecast(train_data, test_data, target_column, order, seasonal_order):
    predictions = []
    rolling_train = train_data.copy()
    for date in test_data.index:
        rolling_train.index.freq = 'QS'
        model = sm.tsa.SARIMAX(
            np.log(rolling_train[target_column] + 1),
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False, maxiter=1000, method='powell')
        forecast_log = results.get_forecast(steps=1).predicted_mean
        forecast = np.exp(forecast_log) - 1
        predictions.append(forecast.iloc[0])
        rolling_train = pd.concat([rolling_train, test_data.loc[[date]]])
    return pd.Series(predictions, index=test_data.index)

# %%
# 8. Manual Hyperparameter Tuning Block
# Manual Parameters for Q1
manual_order_q1 = (1, 1, 1)
manual_seasonal_order_q1 = (1, 1, 0, 4)

# Manual Parameters for Q2
manual_order_q2 = (1, 1, 1)
manual_seasonal_order_q2 = (1, 1, 0, 4)

# %%
# 9. Validate on 2022
y_pred_2022_q1 = rolling_forecast(train_df, val_df[val_df['Quarter'] == 1], target_column, manual_order_q1, manual_seasonal_order_q1)
y_pred_2022_q2 = rolling_forecast(train_df, val_df[val_df['Quarter'] == 2], target_column, manual_order_q2, manual_seasonal_order_q2)

mae_q1_2022 = mean_absolute_error(y_val[val_df['Quarter'] == 1], y_pred_2022_q1)
mae_q2_2022 = mean_absolute_error(y_val[val_df['Quarter'] == 2], y_pred_2022_q2)

print(f"Validation MAE for Q1 2022: {mae_q1_2022:.4f}")
print(f"Validation MAE for Q2 2022: {mae_q2_2022:.4f}")

# %%

# 10. Test on 2023
# Ensure that the index has a consistent frequency
train_val_df = pd.concat([train_df, val_df])
train_val_df.index = pd.date_range(start=train_val_df.index.min(), end=train_val_df.index.max(), freq='QS')

# Q1 Predictions for 2023
y_pred_2023_q1 = rolling_forecast(
    train_val_df,
    test_df[test_df['Quarter'] == 1],
    target_column,
    manual_order_q1,
    manual_seasonal_order_q1
)

# Q2 Predictions for 2023
y_pred_2023_q2 = rolling_forecast(
    train_val_df,
    test_df[test_df['Quarter'] == 2],
    target_column,
    manual_order_q2,
    manual_seasonal_order_q2
)

# Calculate MAE for Q1 and Q2
mae_q1_2023 = mean_absolute_error(
    y_test[test_df['Quarter'] == 1],
    y_pred_2023_q1
)

mae_q2_2023 = mean_absolute_error(
    y_test[test_df['Quarter'] == 2],
    y_pred_2023_q2
)

# Display MAE values
print(f"Test MAE for Q1 2023: {mae_q1_2023:.4f}")
print(f"Test MAE for Q2 2023: {mae_q2_2023:.4f}")


# 11. Forecast for 2024
y_pred_2024_q1 = rolling_forecast(pd.concat([train_df, val_df, test_df[test_df['Year'] == 2023]]), test_df[test_df['Quarter'] == 1], target_column, manual_order_q1, manual_seasonal_order_q1)
y_pred_2024_q2 = rolling_forecast(pd.concat([train_df, val_df, test_df[test_df['Year'] == 2023]]), test_df[test_df['Quarter'] == 2], target_column, manual_order_q2, manual_seasonal_order_q2)

print(f"Forecast for Q1 2024: {y_pred_2024_q1.values}")
print(f"Forecast for Q2 2024: {y_pred_2024_q2.values}")
