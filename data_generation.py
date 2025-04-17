import numpy as np
import pandas as pd

def generate_synthetic_data(num_skus=10, n_months=24, seed=42):
    """
    Generates synthetic monthly 'Actual' data for `num_skus` SKUs over `n_months`.
    
    Returns a DataFrame with columns:
      SKU, Actual_Month, Actual_Value
    """
    np.random.seed(seed)
    rows = []
    skus = [f"SKU_{i+1}" for i in range(num_skus)]
    
    for sku in skus:
        # Random baseline, trend, and seasonality
        baseline = 100 + np.random.normal(0, 10)
        trend = np.random.normal(0, 1)
        seasonality = np.random.normal(0, 5, size=n_months)
        
        for m_idx in range(n_months):
            month = m_idx + 1
            # Synthetic actual: baseline + (trend * month) + seasonal noise
            val = baseline + (trend * month) + seasonality[m_idx]
            val = max(0, val)  # enforce non-negative
            rows.append((sku, month, val))
            
    df_actual = pd.DataFrame(rows, columns=["SKU", "Actual_Month", "Actual_Value"])
    return df_actual


def generate_forecasts(df_actual, forecast_horizon=12, forecast_noise_std=5.0, ci_factor=1.96):
    """
    For each SKU and Actual_Month in `df_actual`, generate up to `forecast_horizon` months of forecasts.
    Also generates naive forecast intervals (lower & upper) for PICP demonstration.
    
    Returns a DataFrame with columns:
      [SKU, Actual_Month, Actual_Value, Prediction_Month, Prediction_Value, 
       Prediction_Actual, Forecast_Lower, Forecast_Upper]
    """
    prediction_rows = []
    
    # Group by SKU so we handle each SKU separately
    for sku, group in df_actual.groupby("SKU"):
        # Sort by month to ensure correct chronological order
        group_sorted = group.sort_values("Actual_Month")
        months = group_sorted["Actual_Month"].values
        actual_values = group_sorted["Actual_Value"].values
        
        # For quick lookup of future actual
        month_to_value = dict(zip(months, actual_values))
        
        # For each month, forecast up to `forecast_horizon` months ahead
        for i, row in group_sorted.iterrows():
            current_month = row["Actual_Month"]
            current_value = row["Actual_Value"]
            
            for step_ahead in range(1, forecast_horizon + 1):
                pred_month = current_month + step_ahead
                # Only forecast if pred_month is within our actual data range
                if pred_month in month_to_value:
                    future_actual_val = month_to_value[pred_month]
                    
                    # Naive forecast: future_actual_val + random noise
                    noise = np.random.normal(0, forecast_noise_std)
                    forecast_val = future_actual_val + noise
                    
                    # Confidence interval (for demonstration of PICP)
                    lower_bound = forecast_val - (ci_factor * forecast_noise_std)
                    upper_bound = forecast_val + (ci_factor * forecast_noise_std)
                    
                    prediction_rows.append([
                        sku,
                        current_month,
                        current_value,
                        pred_month,
                        forecast_val,
                        future_actual_val,
                        lower_bound,
                        upper_bound
                    ])
                
    df_pred = pd.DataFrame(prediction_rows, columns=[
        "SKU",
        "Actual_Month",
        "Actual_Value",
        "Prediction_Month",
        "Prediction_Value",
        "Prediction_Actual",
        "Forecast_Lower",
        "Forecast_Upper"
    ])
    return df_pred