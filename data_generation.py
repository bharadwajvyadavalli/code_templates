import numpy as np
import pandas as pd

def generate_date_list(start_ym="2025-01", n_months=24):
    """
    Creates a list of YYYY-MM strings starting from `start_ym` for `n_months`.
    Example: start_ym = '2025-01', n_months=24 -> '2025-01', '2025-02', ... '2026-12'.
    """
    start_date = pd.to_datetime(start_ym, format="%Y-%m")
    date_range = pd.date_range(start=start_date, periods=n_months, freq='MS')
    # Convert each to YYYY-MM string
    return [dt.strftime("%Y-%m") for dt in date_range]

def generate_baseline_data(num_skus=10, n_months=24, start_ym="2025-01", seed=42):
    """
    1) Generate synthetic 'Actual' data for `num_skus` SKUs over a monthly range 
       of length `n_months`, starting from `start_ym`.
    2) For each (SKU, Actual_Month), create up to 12 months of naive forecasts
       (only if that future month is in the 24-month window).

    Returns
    -------
    df_pred : pd.DataFrame
      Columns:
        SKU
        Actual_Month        # 'YYYY-MM' format
        Actual_Value
        Prediction_Month    # 'YYYY-MM' format
        Prediction_Value
        Prediction_Actual
      (Floats at this stage, no anomalies injected yet.)
    """
    np.random.seed(seed)
    
    skus = [f"SKU_{i+1}" for i in range(num_skus)]
    
    # Build the list of months from 2025-01 up to 24 months ahead (through 2026-12)
    all_months = generate_date_list(start_ym, n_months)
    
    # Make a DataFrame of actual data
    actual_rows = []
    for sku in skus:
        # Each SKU has a random baseline and trend
        baseline = 100 + np.random.normal(0, 10)
        trend = np.random.normal(0, 2)
        # We'll add a bit of random noise per month
        for i, ym in enumerate(all_months):
            month_idx = i + 1  # 1..24
            val = baseline + trend*month_idx + np.random.normal(0, 5)
            val = max(val, 0)  # ensure non-negative
            actual_rows.append((sku, ym, val))
    
    df_actual = pd.DataFrame(actual_rows, columns=["SKU", "Actual_Month", "Actual_Value"])
    
    # Convert 'YYYY-MM' to a datetime for easy offset checks
    df_actual["dt"] = pd.to_datetime(df_actual["Actual_Month"], format="%Y-%m")
    
    # Create naive forecasts
    forecast_rows = []
    for sku in skus:
        sub = df_actual[df_actual["SKU"] == sku].copy().sort_values("dt")
        # Build a lookup of dt -> Actual_Value
        dt_to_val = dict(zip(sub["dt"], sub["Actual_Value"]))
        
        all_dt = sorted(sub["dt"].unique())
        # We'll convert the list of all datetimes into a set for quick membership
        dt_set = set(all_dt)
        
        for row in sub.itertuples(index=False):
            origin_dt = row.dt
            origin_val = row.Actual_Value
            # We'll attempt up to 12 months ahead
            for m_ahead in range(1, 13):
                future_dt = origin_dt + pd.DateOffset(months=m_ahead)
                if future_dt in dt_set:
                    future_val = dt_to_val[future_dt]
                    # naive forecast = future_val + small random noise
                    pred_val = future_val + np.random.normal(0, 5)
                    
                    forecast_rows.append([
                        sku,
                        row.Actual_Month,      # e.g. '2025-01'
                        origin_val,
                        future_dt.strftime("%Y-%m"),  # e.g. '2025-02'
                        pred_val,
                        future_val
                    ])
    
    df_pred = pd.DataFrame(forecast_rows, columns=[
        "SKU",
        "Actual_Month",
        "Actual_Value",
        "Prediction_Month",
        "Prediction_Value",
        "Prediction_Actual"
    ])
    
    return df_pred


def inject_anomalies_and_bias(
    df_pred,
    data_anomaly_prob=0.05,
    data_anomaly_factor=5.0,
    residual_anomaly_prob=0.05,
    residual_anomaly_factor=30.0,
    bias_sku_list=("SKU_1", "SKU_2"),
    bias_month_range=("2025-10", "2025-12"),  # or ("2025-10", "2026-02") etc.
    bias_shift=25.0,
    seed=999
):
    """
    Inject anomalies & systematic bias spikes into df_pred:
    
    1) Data anomalies: for ~data_anomaly_prob fraction of rows, 
       multiply or divide 'Prediction_Actual' by data_anomaly_factor 
       to simulate extreme actuals.
    2) Residual anomalies: for ~residual_anomaly_prob fraction of rows,
       add +/- residual_anomaly_factor to 'Prediction_Value'.
    3) Systematic bias: for the specified SKU(s) and 'Actual_Month' in [bias_month_range],
       add bias_shift to 'Prediction_Value'.

    Parameters
    ----------
    df_pred : DataFrame
      Must have columns [SKU, Actual_Month, Actual_Value, 
                        Prediction_Month, Prediction_Value, Prediction_Actual].
    data_anomaly_prob : float
      Probability of picking a row to create a data anomaly in 'Prediction_Actual'.
    data_anomaly_factor : float
      Factor for extremes (multiplying or dividing).
    residual_anomaly_prob : float
      Probability of picking a row to create a large forecast error in 'Prediction_Value'.
    residual_anomaly_factor : float
      Magnitude of that forced forecast error.
    bias_sku_list : tuple or list of strings
      SKUs to which we apply a systematic bias shift.
    bias_month_range : (str, str)
      The range of origin months (in 'YYYY-MM' format, inclusive) for applying the bias shift.
      e.g. ("2025-10", "2025-12") means if Actual_Month is in [2025-10, 2025-12], 
      we add a shift to Prediction_Value.
    bias_shift : float
      How much to add to 'Prediction_Value' to create the bias.
    seed : int
      Random seed for reproducibility.
    
    Returns
    -------
    df_pred_aug : DataFrame
      Contains forced anomalies and bias spikes (still floats at this point).
    """
    np.random.seed(seed)
    df = df_pred.copy()
    
    # 1) Data anomalies
    mask_data_anom = np.random.rand(len(df)) < data_anomaly_prob
    for i in df[mask_data_anom].index:
        # 50% get multiplied, 50% get divided
        if np.random.rand() < 0.5:
            df.at[i, "Prediction_Actual"] *= data_anomaly_factor
        else:
            df.at[i, "Prediction_Actual"] /= data_anomaly_factor
    
    # 2) Residual anomalies
    mask_res_anom = np.random.rand(len(df)) < residual_anomaly_prob
    signs = np.random.choice([1, -1], size=mask_res_anom.sum())
    df.loc[mask_res_anom, "Prediction_Value"] += signs * residual_anomaly_factor
    
    # 3) Systematic bias for certain SKUs and origin months in bias_month_range
    # We'll interpret the bias_month_range as inclusive. 
    # Convert 'YYYY-MM' to datetime for comparison.
    df["AM_dt"] = pd.to_datetime(df["Actual_Month"], format="%Y-%m")
    start_dt = pd.to_datetime(bias_month_range[0], format="%Y-%m")
    end_dt   = pd.to_datetime(bias_month_range[1], format="%Y-%m")
    
    for idx, row in df.iterrows():
        if row["SKU"] in bias_sku_list:
            if start_dt <= row["AM_dt"] <= end_dt:
                df.at[idx, "Prediction_Value"] += bias_shift
    
    # Drop helper column
    df.drop(columns="AM_dt", inplace=True)
    return df


def main():
    # 1) Generate baseline data for 10 SKUs, from '2025-01' for 24 months (through '2026-12')
    df_pred = generate_baseline_data(
        num_skus=10, 
        n_months=24, 
        start_ym="2025-01",
        seed=42
    )
    
    # 2) Inject anomalies & bias
    df_pred_aug = inject_anomalies_and_bias(
        df_pred,
        data_anomaly_prob=0.08,       # 8% data anomalies
        data_anomaly_factor=5.0,
        residual_anomaly_prob=0.10,   # 10% big forecast errors
        residual_anomaly_factor=30.0,
        bias_sku_list=["SKU_1", "SKU_2"],
        bias_month_range=("2025-10", "2026-02"), # apply bias from '2025-10' to '2026-02'
        bias_shift=25.0,
        seed=999
    )
    
    # 3) Convert all numeric columns to integers
    df_pred_aug["Actual_Value"] = np.round(df_pred_aug["Actual_Value"]).astype(int)
    df_pred_aug["Prediction_Value"] = np.round(df_pred_aug["Prediction_Value"]).astype(int)
    df_pred_aug["Prediction_Actual"] = np.round(df_pred_aug["Prediction_Actual"]).astype(int)
    
    print("\n=== FINAL df_pred_aug (YYYY-MM format, INT VALUES, with forced anomalies & bias) ===")
    print(df_pred_aug.head(20))
    
    # Optionally, save to CSV
    # df_pred_aug.to_csv("sample_data_with_anomalies_int.csv", index=False)

if __name__ == "__main__":
    main()