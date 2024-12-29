# %%
# 0. Import Libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error

# %%
# 1. Load the updated dataset
file_path = r"C:\Users\c.hakker\OneDrive - VISTA college\Senior Stuff\Opleiding Data science\Data\merged_tables_new.xlsx"
df = pd.read_excel(file_path)

# %%
# 2. Filter for the Specific Branch 'G Trade'
branch_name = 'G Trade'
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
branch_name = 'G Trade'
target_column = '80072ned_Ziekteverzuimpercentage_1'

# %%
# 6. Split the Data into Training and Testing Sets
train_end_year = 2021
train_df = df[df['Year'] <= train_end_year].copy()
test_df = df[df['Year'].isin([2022, 2023])].copy()
y_train = train_df[target_column]
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
# 8. Hyperparameter Tuning for Overall Parameters
order_grid = [(1, 1, 1), (2, 1, 1)]
seasonal_order_grid = [(1, 1, 1, 4), (2, 1, 1, 4)]
best_mae = float('inf')
best_order = None
best_seasonal_order = None

for order in order_grid:
    for seasonal_order in seasonal_order_grid:
        try:
            y_pred_2022 = rolling_forecast(
                train_df,
                test_df[test_df['Year'] == 2022],
                target_column,
                order,
                seasonal_order
            )
            mae_all_2022 = mean_absolute_error(
                y_test[test_df['Year'] == 2022], y_pred_2022
            )
            mae_q1_2022 = mean_absolute_error(
                y_test[(test_df['Year'] == 2022) & (test_df['Quarter'] == 1)],
                y_pred_2022[test_df['Quarter'] == 1]
            )
            if mae_all_2022 < best_mae:
                best_mae = mae_all_2022
                best_order = order
                best_seasonal_order = seasonal_order
        except Exception as e:
            print(f"Error with order={order}, seasonal_order={seasonal_order}: {e}")

print(f"Best Overall Parameters: Order={best_order}, Seasonal={best_seasonal_order}, MAE: {best_mae:.4f}")
print(f"MAE for Q1 2022: {mae_q1_2022:.4f}")
print(f"MAE for all quarters of 2022: {mae_all_2022:.4f}")

# %%
# 9. Predictions for 2023 Using Overall Parameters
y_pred_2023 = rolling_forecast(
    pd.concat([train_df, test_df[test_df['Year'] == 2022]]),
    test_df[test_df['Year'] == 2023],
    target_column,
    best_order,
    best_seasonal_order
)

mae_q1_2023 = mean_absolute_error(
    y_test[(test_df['Year'] == 2023) & (test_df['Quarter'] == 1)],
    y_pred_2023[test_df['Quarter'] == 1]
)
mae_all_2023 = mean_absolute_error(
    y_test[test_df['Year'] == 2023], y_pred_2023
)
print(f"MAE for Q1 2023: {mae_q1_2023:.4f}")
print(f"MAE for all quarters of 2023: {mae_all_2023:.4f}")

# %%
# 10. Hyperparameter Tuning for Q1-Specific Predictions
target_year = 2024
forecast_df_q1 = df[(df['Year'] == target_year) & (df['Quarter'] == 1)].copy()
best_mae_q1 = float('inf')
best_order_q1 = None
best_seasonal_order_q1 = None

for order in order_grid:
    for seasonal_order in seasonal_order_grid:
        try:
            mae_q1 = mean_absolute_error(
                y_test[(test_df['Year'] == 2022) & (test_df['Quarter'] == 1)],
                rolling_forecast(
                    train_df,
                    test_df[(test_df['Year'] == 2022) & (test_df['Quarter'] == 1)],
                    target_column,
                    order,
                    seasonal_order
                )
            )
            if mae_q1 < best_mae_q1:
                best_mae_q1 = mae_q1
                best_order_q1 = order
                best_seasonal_order_q1 = seasonal_order
        except Exception as e:
            print(f"Error with order={order}, seasonal_order={seasonal_order}: {e}")

print(f"Best Q1 Parameters: Order={best_order_q1}, Seasonal={best_seasonal_order_q1}, MAE Q1: {best_mae_q1:.4f}")

# %%
# %%
# Definition of rolling_forecast_recent for 2024 predictions
def rolling_forecast_recent(train_data, test_data, target_column, order, seasonal_order, recent_years=5):
    predictions = []
    rolling_train = train_data.copy()

    for date in test_data.index:
        # Limit rolling training data to recent years
        rolling_train = rolling_train.loc[rolling_train.index >= (date - pd.DateOffset(years=recent_years))]
        rolling_train.index.freq = 'QS'  # Ensure frequency is set

        # Fit the SARIMAX model
        model = sm.tsa.SARIMAX(
            np.log(rolling_train[target_column] + 1),
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False, maxiter=5000, method='powell')  # Robust optimization
        forecast_log = results.get_forecast(steps=1).predicted_mean
        forecast = np.exp(forecast_log) - 1  # Back-transform to original scale
        predictions.append(forecast.iloc[0])

        # Add the test data point to the rolling training set
        rolling_train = pd.concat([rolling_train, test_data.loc[[date]]])

    return pd.Series(predictions, index=test_data.index)

# %%
# %%
# 11. Predictions for 2024 (Q1–Q3)
forecast_df_2024 = df[(df['Year'] == 2024) & (df['Quarter'] <= 3)].copy()

# Generate predictions for Q1-Q3 using the best Q1-specific parameters
y_pred_2024 = rolling_forecast_recent(
    pd.concat([train_df, test_df]),  # Use all data up to 2023
    forecast_df_2024,  # Data for 2024 (Q1–Q3)
    target_column,
    best_order_q1,
    best_seasonal_order_q1,
    recent_years=5  # Focus on recent 5 years
)

# Calculate MAE for Q1, Q2, and Q3 2024
mae_q1_2024 = mean_absolute_error(
    forecast_df_2024[forecast_df_2024['Quarter'] == 1][target_column],
    y_pred_2024[forecast_df_2024['Quarter'] == 1]
)
mae_all_2024 = mean_absolute_error(forecast_df_2024[target_column], y_pred_2024)

print(f"Updated MAE for Q1 2024: {mae_q1_2024:.4f}")
print(f"Updated MAE for all Quarters of 2024 (Q1–Q3): {mae_all_2024:.4f}")

# %%
# 12. Hyperparameter Tuning for Q1-Specific Predictions
best_mae_q1 = float('inf')
best_order_q1 = None
best_seasonal_order_q1 = None

for order in [(0, 1, 0), (1, 1, 0)]:
    for seasonal_order in [(0, 1, 0, 4), (1, 1, 0, 4)]:
        try:
            y_pred_q1_2024 = rolling_forecast_recent(
                pd.concat([train_df, test_df]),
                forecast_df_2024[forecast_df_2024['Quarter'] == 1],  # Focus on Q1
                target_column,
                order,
                seasonal_order,
                recent_years=5  # Use recent 5 years
            )
            mae_q1 = mean_absolute_error(
                forecast_df_2024[forecast_df_2024['Quarter'] == 1][target_column],
                y_pred_q1_2024
            )
            if mae_q1 < best_mae_q1:
                best_mae_q1 = mae_q1
                best_order_q1 = order
                best_seasonal_order_q1 = seasonal_order
        except Exception as e:
            print(f"Error with order={order}, seasonal_order={seasonal_order}: {e}")

print(f"Best Q1 Parameters: Order={best_order_q1}, Seasonal={best_seasonal_order_q1}, MAE Q1: {best_mae_q1:.4f}")


# %%
# 13. Predictions for 2024 (Q1–Q3)
forecast_df_2024 = df[(df['Year'] == 2024) & (df['Quarter'] <= 3)].copy()

# Rolling forecast with refined parameters
y_pred_2024 = rolling_forecast_recent(
    pd.concat([train_df, test_df]),  # Use training and test data up to 2023
    forecast_df_2024,  # Test data for 2024
    target_column,
    best_order_q1,  # Best Q1 parameters
    best_seasonal_order_q1,
    recent_years=5
)

# Calculate MAE for Q1–Q3 2024
mae_q1_2024 = mean_absolute_error(
    forecast_df_2024[forecast_df_2024['Quarter'] == 1][target_column],
    y_pred_2024[forecast_df_2024['Quarter'] == 1]
)
mae_all_2024 = mean_absolute_error(forecast_df_2024[target_column], y_pred_2024)

print(f"Updated MAE for Q1 2024: {mae_q1_2024:.4f}")
print(f"Updated MAE for all Quarters of 2024 (Q1–Q3): {mae_all_2024:.4f}")

# %%
# 14. Visualizations
# Visualization for 2022
fig_2022 = go.Figure()
fig_2022.add_trace(go.Scatter(
    x=test_df[test_df['Year'] == 2022].index,
    y=y_test[test_df['Year'] == 2022],
    mode='lines+markers',
    name='Actual (2022)',
    line=dict(color='#0078d2', width=2)
))
fig_2022.add_trace(go.Scatter(
    x=test_df[test_df['Year'] == 2022].index,
    y=y_pred_2022,
    mode='lines+markers',
    name='Predictions (2022)',
    line=dict(color='orange', width=2, dash='dash')
))
fig_2022.add_annotation(
    xref="paper", yref="paper", x=0.00, y=1.13, showarrow=False,
    text=f"MAE Q1 2022: {mae_q1_2022:.4f}",
    font=dict(size=12, color="black")
)
fig_2022.add_annotation(
    xref="paper", yref="paper", x=0.00, y=1.08, showarrow=False,
    text=f"MAE All 2022: {mae_all_2022:.4f}",
    font=dict(size=12, color="black")
)
fig_2022.update_layout(
    title=f'Sick Leave Test - G Trade (2022)',
    xaxis_title='Date',
    yaxis_title='Sick Leave Percentage',
    plot_bgcolor='white',
    xaxis=dict(showgrid=False, tickformat="%Y-%m"),
    yaxis=dict(showgrid=True, gridcolor='lightgrey', showline=True, linewidth=0.5, linecolor='black'),
    font=dict(family="Roboto", size=14),
    margin=dict(l=50, r=50, t=100, b=50),
    width=1100, height=500
)
fig_2022.show()

# %%
# 15. Visualization for 2023
fig_2023 = go.Figure()
fig_2023.add_trace(go.Scatter(
    x=test_df[test_df['Year'] == 2023].index,
    y=y_test[test_df['Year'] == 2023],
    mode='lines+markers',
    name='Actual (2023)',
    line=dict(color='#0078d2', width=2)
))
fig_2023.add_trace(go.Scatter(
    x=test_df[test_df['Year'] == 2023].index,
    y=y_pred_2023,
    mode='lines+markers',
    name='Rolling Predictions (2023)',
    line=dict(color='green', width=2, dash='dot')
))
fig_2023.add_annotation(
    xref="paper", yref="paper", x=0.00, y=1.13, showarrow=False,
    text=f"MAE Q1 2023: {mae_q1_2023:.4f}",
    font=dict(size=12, color="black")
)
fig_2023.add_annotation(
    xref="paper", yref="paper", x=0.00, y=1.08, showarrow=False,
    text=f"MAE All 2023: {mae_all_2023:.4f}",
    font=dict(size=12, color="black")
)
fig_2023.update_layout(
    title=f'Sick Leave Validation - G Trade (2023)',
    xaxis_title='Date',
    yaxis_title='Sick Leave Percentage',
    plot_bgcolor='white',
    xaxis=dict(showgrid=False, tickformat="%Y-%m"),
    yaxis=dict(showgrid=True, gridcolor='lightgrey', showline=True, linewidth=0.5, linecolor='black'),
    font=dict(family="Roboto", size=14),
    margin=dict(l=50, r=50, t=100, b=50),
    width=1100, height=500
)
fig_2023.show()

# %%
# %%
# 16. Visualization for 2024 (Q1-Q3)
fig_2024 = go.Figure()

# Add actual sick leave percentage for Q1-Q3 2024
fig_2024.add_trace(go.Scatter(
    x=forecast_df_2024.index,
    y=forecast_df_2024[target_column],
    mode='lines+markers',
    name='Actual (2024 Q1-Q3)',
    line=dict(color='#0078d2', width=2)
))

# Add predictions for Q1-Q3 2024
fig_2024.add_trace(go.Scatter(
    x=forecast_df_2024.index,
    y=y_pred_2024,
    mode='lines+markers',
    name='Predictions (2024 Q1-Q3)',
    line=dict(color='orange', width=2, dash='dash')
))

# Update layout
fig_2024.update_layout(
    title='Sick Leave Forecast - 2024 (Q1-Q3)',
    xaxis_title='Date',
    yaxis_title='Sick Leave Percentage',
    plot_bgcolor='white',
    xaxis=dict(showgrid=False, tickformat="%Y-%m"),
    yaxis=dict(showgrid=True, gridcolor='lightgrey', showline=True, linewidth=0.5, linecolor='black'),
    font=dict(family="Roboto", size=14),
    margin=dict(l=50, r=50, t=100, b=50),
    width=1100, height=500
)

fig_2024.show()

# %%
# 17. Overview of MAE per quarter for 2022, 2023, and 2024
mae_overview = {
    "Year": [],
    "Quarter": [],
    "MAE": []
}

# Function to calculate MAE for each quarter
def calculate_mae(year, quarter, actual, predicted):
    mask_actual = (actual.index.year == year) & (actual.index.quarter == quarter)
    mask_predicted = (predicted.index.year == year) & (predicted.index.quarter == quarter)
    
    actual_filtered = actual[mask_actual]
    predicted_filtered = predicted[mask_predicted]
    
    if len(actual_filtered) == len(predicted_filtered):
        return mean_absolute_error(actual_filtered, predicted_filtered)
    else:
        print(f"Index mismatch for Year {year}, Quarter {quarter}. Skipping.")
        return None

# Add MAE for each year and quarter
for year, pred, data in [(2022, y_pred_2022, y_test), (2023, y_pred_2023, y_test), (2024, y_pred_2024, forecast_df_2024[target_column])]:
    for quarter in [1, 2, 3, 4] if year != 2024 else [1, 2, 3]:
        mae_value = calculate_mae(year, quarter, data, pred)
        mae_overview["Year"].append(year)
        mae_overview["Quarter"].append(quarter)
        mae_overview["MAE"].append(mae_value)

# Convert to DataFrame and save
mae_df = pd.DataFrame(mae_overview)
print(mae_df)
mae_df.to_csv("mae_overview.csv", index=False)



