# data_generation.py

import numpy as np
import pandas as pd

def generate_synthetic_data(num_skus=1, n_months=24, start_ym="2025-01", seed=42):
    """
    Generates synthetic monthly 'Actual' data in the format YYYY-MM, starting from `start_ym`,
    for `num_skus` SKUs over `n_months` months.
    
    - Actual_Month is stored as a string (e.g., '2025-01').
    - Actual_Value is an integer.

    Returns a DataFrame with columns:
      SKU, Actual_Month, Actual_Value
    """
    np.random.seed(seed)
    
    # Create a date range (monthly) from the start_ym
    # freq='MS' => month start. We generate `n_months` periods.
    date_range = pd.date_range(start=start_ym, periods=n_months, freq='MS')
    
    skus = [f"SKU_{i+1}" for i in range(num_skus)]
    
    rows = []
    for sku in skus:
        # For each month in the date range, generate an integer Actual_Value
        # Let's pick values in [70..130] for demonstration.
        actual_values = np.random.randint(low=70, high=131, size=n_months)
        
        for dt, val in zip(date_range, actual_values):
            month_str = dt.strftime("%Y-%m")  # e.g., '2025-01'
            rows.append((sku, month_str, val))
    
    df_actual = pd.DataFrame(rows, columns=["SKU", "Actual_Month", "Actual_Value"])
    return df_actual


def generate_forecasts(df_actual, forecast_horizon=12, forecast_noise_std=5, seed=999):
    """
    Generates forecasts for up to `forecast_horizon` months ahead, 
    using a naive approach:
      - For each (SKU, Actual_Month), find the *future actual* value
        and add a small random noise (integer) to simulate a forecast.
      - Prediction_Month is also in YYYY-MM format.

    Columns returned:
      SKU,
      Actual_Month,
      Actual_Value,
      Prediction_Month,
      Prediction_Value,
      Prediction_Actual

    (All Value columns as integers.)
    """
    np.random.seed(seed)
    
    prediction_rows = []
    
    # Convert Actual_Month strings back to datetime for easier manipulation
    df_actual["Actual_Month_dt"] = pd.to_datetime(df_actual["Actual_Month"], format="%Y-%m")
    
    # Group by SKU so we handle each separately
    for sku, group_sku in df_actual.groupby("SKU"):
        group_sku = group_sku.sort_values("Actual_Month_dt")
        
        # Quick lookup: dt -> (month_str, actual_value)
        dt_to_val = dict(zip(group_sku["Actual_Month_dt"], group_sku["Actual_Value"]))
        
        # For each row in this SKU, forecast up to horizon months
        for row in group_sku.itertuples(index=False):
            current_dt = row.Actual_Month_dt
            current_str = row.Actual_Month
            current_val = row.Actual_Value
            
            for i in range(1, forecast_horizon+1):
                # Future date
                future_dt = current_dt + pd.DateOffset(months=i)
                if future_dt in dt_to_val:
                    future_actual = dt_to_val[future_dt]
                    # Add random noise to create a forecast
                    noise = np.random.normal(0, forecast_noise_std)
                    forecast_val = int(round(future_actual + noise))
                    
                    # Convert future_dt to YYYY-MM
                    future_str = future_dt.strftime("%Y-%m")
                    
                    prediction_rows.append([
                        sku,
                        current_str,
                        current_val,
                        future_str,
                        forecast_val,
                        future_actual
                    ])
    
    df_pred = pd.DataFrame(prediction_rows, columns=[
        "SKU",
        "Actual_Month",
        "Actual_Value",
        "Prediction_Month",
        "Prediction_Value",
        "Prediction_Actual"
    ])
    
    return df_pred


if __name__ == "__main__":
    # EXAMPLE: Show how to generate data for ONE SKU and produce forecasts
    
    # 1) Generate actual data (one SKU, 24 months starting 2025-01)
    df_actual = generate_synthetic_data(num_skus=1, n_months=24, start_ym="2025-01", seed=42)
    print("=== ACTUAL DATA (ONE SKU) ===")
    print(df_actual)
    
    # 2) Generate forecasts (12-month horizon)
    df_pred = generate_forecasts(df_actual, forecast_horizon=12, forecast_noise_std=5, seed=999)
    print("\n=== FORECAST DATA (ONE SKU) ===")
    print(df_pred)