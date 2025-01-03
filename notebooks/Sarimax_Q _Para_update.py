# %%
# 0. Import the libraries
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
# Add a COVID-19 indicator variable to the dataset
df['covid_effect'] = np.where(df['Year'] >= 2022, 1, 0)

# %%
# 2. Filter for the specific branch 'Q Healthcare'
branch_name = 'Q Healthcare'
df = df[df['BedrijfstakkenBranchesSBI2008'].str.strip().str.lower() == branch_name.strip().lower()]

# %%
# 3. Verify required columns
if 'Year' not in df.columns or 'Quarter' not in df.columns:
    raise KeyError("The 'Year' and 'Quarter' columns are required in the dataset.")

# %%
# 4. Create the 'Date' column for quarterly alignment
# Ensure 'Quarter' is an integer
df['Quarter'] = df['Quarter'].astype(int)

# Create the 'Date' column explicitly for the start of each quarter
df['Date'] = pd.to_datetime(
    df['Year'].astype(str) + '-' + (df['Quarter'] * 3 - 2).astype(str) + '-01'
)

# %%
# 5. Set 'Date' as the index
df.set_index('Date', inplace=True)

# Force the index to align with QS-JAN frequency
expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='QS-JAN')

# Reindex the DataFrame explicitly to the expected index
df = df.reindex(expected_index)

# Force frequency metadata explicitly
df.index.freq = 'QS-JAN'

# %%
# 6. Define parameters for SARIMAX model
branch_name = 'Q Healthcare'
target_column = '80072ned_Ziekteverzuimpercentage_1'

# %%
# 8. Define the year ranges
train_end_year = 2021  # Training ends at 2021
validation_year = 2022  # Validation on 2022
test_years = [2023]  # Testing on 2023

# Training data: 2008–2021
train_df = df[df['Year'] <= train_end_year].copy()

# Validation data: 2022
validation_df = df[df['Year'] == validation_year].copy()

# Testing data: 2023
test_df = df[df['Year'].isin(test_years)].copy()

# Define the validation target
y_validation = validation_df[target_column]

# Handle Q1 2022 Outlier
outlier_mask_q1 = (validation_df['Quarter'] == 1)  # Identify Q1 2022
fixed_smooth_value_q1 = 7.65  # Set the fixed smoothed value for Q1

# Replace Q1 2022 values in validation_df
validation_df.loc[outlier_mask_q1, target_column] = fixed_smooth_value_q1

# Handle Q2 2022 Outlier
outlier_mask_q2 = (validation_df['Quarter'] == 2)  # Identify Q2 2022
fixed_smooth_value_q2 = 7.00  # Set the fixed smoothed value for Q2 (adjust as needed)

# Replace Q2 2022 values in validation_df
validation_df.loc[outlier_mask_q2, target_column] = fixed_smooth_value_q2

# Update y_validation to reflect the changes in validation_df
y_validation = validation_df[target_column]

print(f"Flattened Q1 2022 outlier with fixed smoothed value: {fixed_smooth_value_q1}")
print(f"Flattened Q2 2022 outlier with fixed smoothed value: {fixed_smooth_value_q2}")

# Check data preparation
print(f"Training data: {train_df.shape}")
print(f"Validation data: {validation_df.shape}")
print(f"Testing data: {test_df.shape}")

# %%
# 9. Fit SARIMAX on 2008–2021 and validate on 2022
y_train_log = np.log(train_df[target_column] + 1)
y_validation_log = np.log(y_validation + 1)  # Use the updated y_validation

# Fit SARIMAX model on training data (2008–2021)
print("Fitting SARIMAX model for validation on 2022...")
results_2022 = sm.tsa.SARIMAX(
    y_train_log,
    order=(0, 1, 1),  # Adjust these parameters as necessary
    seasonal_order=(1, 1, 0, 4),  # Adjust these parameters as necessary
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

# Forecast for validation (2022)
forecast_2022 = results_2022.get_forecast(steps=len(validation_df))
y_pred_2022_log = forecast_2022.predicted_mean
y_pred_2022 = np.exp(y_pred_2022_log) - 1  # Back-transform predictions to the original scale

# Calculate Mean Absolute Error (MAE) for 2022
mae_q1_2022 = mean_absolute_error(
    y_validation[validation_df['Quarter'] == 1], 
    y_pred_2022[validation_df['Quarter'] == 1]
)
mae_all_2022 = mean_absolute_error(y_validation, y_pred_2022)

print(f"MAE for Q1 2022: {mae_q1_2022:.4f}")
print(f"MAE for all quarters of 2022: {mae_all_2022:.4f}")


# %%
# Include all 2022 data in the rolling training set
train_df_extended = pd.concat([train_df, validation_df])
rolling_train = train_df_extended.copy()

# Set frequency explicitly for all DataFrames
train_df = train_df.asfreq('QS-JAN')
validation_df = validation_df.asfreq('QS-JAN')
test_df = test_df.asfreq('QS-JAN')
rolling_train = rolling_train.asfreq('QS-JAN')

# Initialize a dictionary to store predictions aligned with the test_df index
predictions_dict = {}

# Loop through 2023 test data
for date in test_df[test_df['Year'] == 2023].index:
    # Check if the current date corresponds to Q2
    is_q2 = test_df.loc[date, 'Quarter'] == 2

    # Adjust parameters specifically for Q2
    seasonal_order = (0, 1, 0, 4) if is_q2 else (1, 1, 0, 4)

    # Fit SARIMAX model on the rolling training set
    model_rolling = sm.tsa.SARIMAX(
        np.log(rolling_train[target_column] + 1),
        order=(0, 1, 1),
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        dates=rolling_train.index,  # Explicitly pass the index
        freq='QS-JAN'  # Explicitly set the frequency
    )

    results_rolling = model_rolling.fit(
        disp=False,
        method='powell',
        maxiter=2000,
        xtol=1e-4
    )
    
    # Forecast the next step (1 quarter ahead)
    prediction_log = results_rolling.get_forecast(steps=1).predicted_mean
    prediction = np.exp(prediction_log) - 1  # Back-transform prediction

    # Add the prediction to the dictionary
    predictions_dict[date] = prediction.iloc[0]

    # Update rolling_train with the actual value from the test set
    rolling_train = pd.concat([rolling_train, test_df.loc[[date]]])

# Convert the dictionary to a pandas Series
y_pred_2023 = pd.Series(predictions_dict)

# Verify alignment
print("Predictions Aligned with Index:")
print(y_pred_2023)

# Define y_test
y_test = test_df[target_column]

# Ensure indices align for Q1
aligned_y_test_q1 = y_test[(y_test.index.year == 2023) & (test_df['Quarter'] == 1)]
aligned_y_pred_q1 = y_pred_2023[(y_pred_2023.index.year == 2023) & (test_df['Quarter'] == 1)]

# Calculate MAE for Q1 2023
mae_q1_2023 = mean_absolute_error(aligned_y_test_q1, aligned_y_pred_q1)

# Ensure indices align for all quarters
aligned_y_test_all = y_test[y_test.index.year == 2023]
aligned_y_pred_all = y_pred_2023[y_pred_2023.index.year == 2023]

# Calculate MAE for all quarters of 2023
mae_all_2023 = mean_absolute_error(aligned_y_test_all, aligned_y_pred_all)

print(f"MAE for Q1 2023: {mae_q1_2023:.4f}")
print(f"MAE for all quarters of 2023: {mae_all_2023:.4f}")


# %%
# 11. Forecast for Q1–Q3 2024
# Filter 2024 data for forecasting
forecast_df_2024 = df[(df['Year'] == 2024) & (df['Quarter'] <= 3)].copy()
y_pred_2024 = []  # Predictions for 2024

# Ensure rolling_train is properly initialized and indexed
rolling_train = rolling_train.loc[~rolling_train.index.duplicated(keep='first')].sort_index()

# Loop through the forecast period (Q1–Q3 2024)
for date in forecast_df_2024.index:
    # Ensure rolling_train index has proper frequency
    rolling_train.index = pd.date_range(
        start=rolling_train.index.min(),
        end=rolling_train.index.max(),
        freq='QS'
    )

    # Fit SARIMAX model for 2024 forecasting
    model_forecast = sm.tsa.SARIMAX(
        np.log(rolling_train[target_column] + 1),
        order=(0, 1, 1),
        seasonal_order=(1, 1, 0, 4),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    try:
        # Fit the model
        results_forecast = model_forecast.fit(disp=False, method='powell', maxiter=2000)
        
        # Forecast the next quarter
        prediction_log = results_forecast.get_forecast(steps=1).predicted_mean
        prediction = np.exp(prediction_log) - 1  # Back-transform prediction
        y_pred_2024.append(prediction.iloc[0])
        
        # Update rolling_train with the forecasted date
        rolling_train = pd.concat([rolling_train, forecast_df_2024.loc[[date]]])
    except Exception as e:
        print(f"Error during 2024 forecast at {date}: {e}")
        break

# Convert predictions to a pandas Series
y_pred_2024 = pd.Series(y_pred_2024, index=forecast_df_2024.index, name="Predictions")

# Calculate MAE for Q1–Q3 2024
try:
    mae_all_2024 = mean_absolute_error(
        forecast_df_2024[target_column], 
        y_pred_2024
    )
    print(f"MAE for all quarters of 2024 (Q1–Q3): {mae_all_2024:.4f}")
except Exception as e:
    print(f"Error calculating MAE for 2024: {e}")


# %%
# 11b. Define y_pred_target for Q1–Q3 2024 (actual values from the dataset)
y_pred_target = forecast_df_2024[target_column]

# %%
# 11c. Ensure indices of y_pred_target and y_pred_2024 align
aligned_forecast_df_2024 = forecast_df_2024.loc[
    forecast_df_2024.index.intersection(y_pred_target.index).intersection(y_pred_2024.index)
]
aligned_y_pred_target = y_pred_target.loc[aligned_forecast_df_2024.index]
aligned_y_pred_2024 = y_pred_2024.loc[aligned_forecast_df_2024.index]

# Calculate MAE for all quarters of 2024 (Q1–Q3)
mae_all_2024 = mean_absolute_error(aligned_y_pred_target, aligned_y_pred_2024)

print(f"MAE for all quarters of 2024 (Q1–Q3): {mae_all_2024:.4f}")


# %%
# Define the test dataset for 2022
test_df_2022 = test_df[test_df['Year'] == 2022].copy()

# Ensure that test_df_2022 has the correct frequency
test_df_2022.index.freq = 'QS'

# %%
# 12. Visualization for 2022
fig_2022 = go.Figure()

# Ensure valid indices for actual and predicted values
valid_indices_2022 = validation_df.index.intersection(y_validation.dropna().index).intersection(y_pred_2022.dropna().index)

# Add actual sick leave percentage line for 2022
fig_2022.add_trace(go.Scatter(
    x=valid_indices_2022,
    y=y_validation.loc[valid_indices_2022],
    mode='lines+markers',
    name='Actual (2022)',
    line=dict(color='#0078d2', width=2)
))

# Add predictions line for 2022
fig_2022.add_trace(go.Scatter(
    x=valid_indices_2022,
    y=y_pred_2022.loc[valid_indices_2022],
    mode='lines+markers',
    name='Predictions (2022)',
    line=dict(color='orange', width=2, dash='dash')
))

# Add MAE annotations for 2022
fig_2022.add_annotation(
    xref="paper", yref="paper", x=0.00, y=1.10, showarrow=False,
    text=f"MAE Q1 2022: {mae_q1_2022:.4f}",
    font=dict(size=12, color="black")
)
fig_2022.add_annotation(
    xref="paper", yref="paper", x=0.00, y=1.05, showarrow=False,
    text=f"MAE All 2022: {mae_all_2022:.4f}",
    font=dict(size=12, color="black")
)

fig_2022.update_layout(
    title=f'Sick Leave Test - Q Healthcare (2022)',
    xaxis_title='Date',
    yaxis_title='Sick Leave Percentage',
    plot_bgcolor='white',
    xaxis=dict(
        showgrid=False,
        tickformat="%Y-%m",
        range=[validation_df.index.min(), validation_df.index.max()]
    ),
    yaxis=dict(
        showgrid=True, gridcolor='lightgrey', showline=True, linewidth=0.5, linecolor='black'
    ),
    font=dict(family="Roboto", size=14),
    margin=dict(l=50, r=50, t=100, b=50),
    width=1100, height=500
)

fig_2022.show()

# %%
# Define y_test from test_df
y_test = test_df[target_column]

# %%
# Ensure y_test is defined
y_test = test_df[target_column]

# Define the figure for 2023
fig_2023 = go.Figure()

# Ensure valid indices for 2023
valid_indices_2023 = test_df.index.intersection(y_test.dropna().index).intersection(y_pred_2023.dropna().index)

# Add actual sick leave percentage line for 2023
fig_2023.add_trace(go.Scatter(
    x=valid_indices_2023,
    y=y_test.loc[valid_indices_2023],
    mode='lines+markers',
    name='Actual (2023)',
    line=dict(color='#0078d2', width=2)
))

# Add rolling predictions line for 2023
fig_2023.add_trace(go.Scatter(
    x=valid_indices_2023,
    y=y_pred_2023.loc[valid_indices_2023],
    mode='lines+markers',
    name='Rolling Predictions (2023)',
    line=dict(color='green', width=2, dash='dot')
))

# Add MAE annotations for 2023
fig_2023.add_annotation(
    xref="paper", yref="paper", x=0.00, y=1.10, showarrow=False,
    text=f"MAE Q1 2023: {mae_q1_2023:.4f}",
    font=dict(size=12, color="black")
)
fig_2023.add_annotation(
    xref="paper", yref="paper", x=0.00, y=1.05, showarrow=False,
    text=f"MAE All 2023: {mae_all_2023:.4f}",
    font=dict(size=12, color="black")
)

fig_2023.update_layout(
    title=f'Sick Leave Validation - Q Healthcare (2023)',
    xaxis_title='Date',
    yaxis_title='Sick Leave Percentage',
    plot_bgcolor='white',
    xaxis=dict(
        showgrid=False,
        tickformat="%Y-%m",
        range=[test_df.index.min(), test_df.index.max()]
    ),
    yaxis=dict(
        showgrid=True, gridcolor='lightgrey', showline=True, linewidth=0.5, linecolor='black'
    ),
    font=dict(family="Roboto", size=14),
    margin=dict(l=50, r=50, t=100, b=50),
    width=1100, height=500
)

fig_2023.show()


# %%
# Visualization for Q1–Q3 2024
fig_2024 = go.Figure()

# Add actual sick leave percentage line for Q1–Q3 2024
fig_2024.add_trace(go.Scatter(
    x=forecast_df_2024.index,
    y=forecast_df_2024[target_column],
    mode='lines+markers',
    name='Actual (2024)',
    line=dict(color='#0078d2', width=2)
))

# Add rolling predictions for Q1–Q3 2024
fig_2024.add_trace(go.Scatter(
    x=forecast_df_2024.index,
    y=y_pred_2024,
    mode='lines+markers',
    name='Rolling Predictions (2024)',
    line=dict(color='purple', width=2, dash='dot')
))

# Add MAE annotation for 2024
fig_2024.add_annotation(
    xref="paper", yref="paper", x=0.00, y=1.10, showarrow=False,
    text=f"MAE All 2024 (Q1–Q3): {mae_all_2024:.4f}",
    font=dict(size=12, color="black")
)

# Update layout for the figure
fig_2024.update_layout(
    title=f'Sick Leave Forecast - Q Healthcare (2024 Q1–Q3)',
    xaxis_title='Date',
    yaxis_title='Sick Leave Percentage',
    plot_bgcolor='white',
    xaxis=dict(
        showgrid=False,
        tickformat="%Y-%m",
        range=[forecast_df_2024.index.min(), forecast_df_2024.index.max()]
    ),
    yaxis=dict(
        showgrid=True, gridcolor='lightgrey', showline=True, linewidth=0.5, linecolor='black'
    ),
    font=dict(family="Roboto", size=14),
    margin=dict(l=50, r=50, t=100, b=50),
    width=1100, height=500
)

fig_2024.show()


# %%
# %% Calculate and Display MAE Per Quarter for 2022, 2023, and 2024 (Q1–Q3)

# Initialize a list to store MAE results
mae_results = []

# Loop through each year and quarter
for year, data, predictions in [
    (2022, validation_df, y_pred_2022),
    (2023, test_df, y_pred_2023),
    (2024, forecast_df_2024, y_pred_2024)
]:
    for quarter in range(1, 5):
        if year == 2024 and quarter > 3:
            continue  # Skip Q4 for 2024
        
        # Filter data for the current quarter
        q_data = data[data['Quarter'] == quarter]
        q_predictions = predictions[q_data.index]

        # If no data is available, skip
        if q_data.empty or q_predictions.empty:
            print(f"No data available for Q{quarter} {year}. Skipping.")
            continue

        # Calculate actual values
        actual = q_data[target_column]
        
        # Ensure indices align between actual and predictions
        aligned_actual = actual.loc[q_predictions.index.intersection(actual.index)]
        aligned_predictions = q_predictions.loc[q_predictions.index.intersection(actual.index)]

        # Calculate MAE
        mae = mean_absolute_error(aligned_actual, aligned_predictions)
        mae_results.append({"Year": year, "Quarter": quarter, "MAE": mae})
        print(f"MAE for Q{quarter} {year}: {mae:.4f}")

# Convert results to a DataFrame for easy viewing
mae_df = pd.DataFrame(mae_results)

# Display the MAE results
print("\nQuarterly MAE Results:")




