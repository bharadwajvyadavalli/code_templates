import pandas as pd
import numpy as np
from scipy.stats import pearsonr, entropy

def jensen_shannon_distance(p, q):
    """
    p, q: discrete probability distributions (arrays summing to 1).
    JS distance = sqrt(0.5 * KL(p||m) + 0.5 * KL(q||m)) with m=(p+q)/2.
    """
    p = np.asarray(p)
    q = np.asarray(q)
    m = 0.5 * (p + q)
    js_divergence = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
    return np.sqrt(js_divergence)

def compute_metrics_by_sku(df_pred, data_anomaly_range=(10, 200), anomaly_std_threshold=3.0):
    """
    Compute forecast metrics on a *per-SKU* basis.
    
    Expected columns in df_pred:
      - SKU
      - Actual_Month
      - Actual_Value
      - Prediction_Month
      - Prediction_Value
      - Prediction_Actual
      (optional) Forecast_Lower
      (optional) Forecast_Upper
    
    Returns
    -------
    df_pred_aug : pd.DataFrame
        df_pred with extra columns: 'Residual', 'Anomaly_Flag', 'Data_Anomaly_Flag', 'Within_Interval' (if intervals exist)
    df_sku_metrics : pd.DataFrame
        Per-SKU metrics table with columns:
          [SKU, Bias, Anomaly_Rate, PICP, Tracking_Signal, Direction_Accuracy,
           Positive_Residual_Count, Negative_Residual_Count, Pearson_Correlation,
           R_squared, Jensen_Shannon_Distance]
    """
    # Ensure the required columns exist
    required_cols = ["SKU", "Actual_Value", "Prediction_Value", "Prediction_Actual"]
    for col in required_cols:
        if col not in df_pred.columns:
            raise ValueError(f"Column '{col}' is missing from df_pred.")
    
    # 1) Compute Residual
    df_pred["Residual"] = df_pred["Prediction_Value"] - df_pred["Prediction_Actual"]
    
    # 2) Data Anomaly Flag
    low_range, high_range = data_anomaly_range
    df_pred["Data_Anomaly_Flag"] = df_pred["Actual_Value"].apply(
        lambda x: 1 if (x < low_range or x > high_range) else 0
    )
    
    # 3) Within Interval for PICP (only if we have Forecast_Lower & Forecast_Upper)
    if "Forecast_Lower" in df_pred.columns and "Forecast_Upper" in df_pred.columns:
        df_pred["Within_Interval"] = df_pred.apply(
            lambda row: 1 if (row["Prediction_Actual"] >= row["Forecast_Lower"] and
                              row["Prediction_Actual"] <= row["Forecast_Upper"]) else 0,
            axis=1
        )
    else:
        # No intervals available; set to NaN or 0
        df_pred["Within_Interval"] = np.nan
    
    # Prepare a list for SKU-level metrics
    sku_list = df_pred["SKU"].unique()
    metrics_rows = []
    
    for sku in sku_list:
        sub = df_pred[df_pred["SKU"] == sku].copy()
        
        # 4) Residual-based anomaly detection
        res_std = sub["Residual"].std()
        if pd.isna(res_std) or res_std == 0:
            sub["Anomaly_Flag"] = 0
        else:
            upper_bound = anomaly_std_threshold * res_std
            lower_bound = -anomaly_std_threshold * res_std
            sub["Anomaly_Flag"] = sub["Residual"].apply(
                lambda x: 1 if x < lower_bound or x > upper_bound else 0
            )
        
        # Assign back to main df_pred
        df_pred.loc[sub.index, "Anomaly_Flag"] = sub["Anomaly_Flag"]
        
        # --- Compute per-SKU metrics ---
        bias = sub["Residual"].mean()
        anomaly_rate = sub["Anomaly_Flag"].mean() if len(sub) > 0 else np.nan
        
        # PICP
        if "Within_Interval" in sub.columns and sub["Within_Interval"].notna().any():
            # If we do have intervals, compute coverage among the non-NaN
            picp_valid = sub["Within_Interval"].dropna()
            picp = picp_valid.mean() if len(picp_valid) > 0 else np.nan
        else:
            picp = np.nan
        
        # Tracking Signal
        residual_sum = sub["Residual"].sum()
        mad = sub["Residual"].abs().mean()
        if mad == 0 or len(sub) == 0:
            tracking_signal = np.nan
        else:
            tracking_signal = residual_sum / (mad * len(sub))
        
        # Direction Accuracy (check step_ahead=1 if feasible)
        direction_checks = []
        # If Actual_Month/Prediction_Month are in numeric or date form, 
        # adjust logic accordingly. For now, let's do a naive approach:
        # We'll parse them as dates if they look like "YYYY-MM".
        try:
            sub["AM_dt"] = pd.to_datetime(sub["Actual_Month"], format="%Y-%m")
            sub["PM_dt"] = pd.to_datetime(sub["Prediction_Month"], format="%Y-%m")
            for row in sub.itertuples(index=False):
                # If the predicted month is exactly 1 month after actual
                if (row.PM_dt.year == row.AM_dt.year and row.PM_dt.month == row.AM_dt.month + 1) \
                   or (row.PM_dt.year == row.AM_dt.year + 1 and row.PM_dt.month == (row.AM_dt.month - 11)):
                    actual_change = row.Prediction_Actual - row.Actual_Value
                    forecast_change = row.Prediction_Value - row.Actual_Value
                    direction_checks.append(np.sign(actual_change) == np.sign(forecast_change))
            direction_accuracy = sum(direction_checks)/len(direction_checks) if len(direction_checks)>0 else np.nan
        except:
            # If there's any date-parsing issue, fallback:
            direction_accuracy = np.nan
        
        # Residual counts
        positive_residual_count = (sub["Residual"] > 0).sum()
        negative_residual_count = (sub["Residual"] < 0).sum()
        
        # Pearson correlation
        if len(sub) > 1:
            pearson_corr, _ = pearsonr(sub["Prediction_Value"], sub["Prediction_Actual"])
        else:
            pearson_corr = np.nan
        
        # R-squared
        y_true = sub["Prediction_Actual"]
        y_pred = sub["Prediction_Value"]
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        if ss_tot == 0:
            r_squared = np.nan
        else:
            r_squared = 1 - (ss_res / ss_tot)
        
        # Jensen-Shannon Distance
        if len(sub) > 1:
            val_min = min(y_true.min(), y_pred.min())
            val_max = max(y_true.max(), y_pred.max())
            if val_min == val_max:
                js_dist = 0.0
            else:
                bins = np.linspace(val_min, val_max, 10)
                true_hist, _ = np.histogram(y_true, bins=bins)
                pred_hist, _ = np.histogram(y_pred, bins=bins)
                if true_hist.sum() == 0 or pred_hist.sum() == 0:
                    js_dist = np.nan
                else:
                    true_prob = true_hist / true_hist.sum()
                    pred_prob = pred_hist / pred_hist.sum()
                    js_dist = jensen_shannon_distance(true_prob, pred_prob)
        else:
            js_dist = np.nan
        
        metrics_rows.append({
            "SKU": sku,
            "Bias": bias,
            "Anomaly_Rate": anomaly_rate,
            "PICP": picp,
            "Tracking_Signal": tracking_signal,
            "Direction_Accuracy": direction_accuracy,
            "Positive_Residual_Count": positive_residual_count,
            "Negative_Residual_Count": negative_residual_count,
            "Pearson_Correlation": pearson_corr,
            "R_squared": r_squared,
            "Jensen_Shannon_Distance": js_dist
        })
    
    df_sku_metrics = pd.DataFrame(metrics_rows)
    return df_pred, df_sku_metrics


if __name__ == "__main__":
    # 1) Read the CSV file (adjust filename/path as needed)
    csv_file = "sample_data_generated.csv"
    df_pred_input = pd.read_csv(csv_file)
    
    # 2) Compute the metrics
    # If you want to define a specific anomaly range or threshold, do so here:
    df_pred_aug, df_sku_metrics = compute_metrics_by_sku(
        df_pred_input,
        data_anomaly_range=(10, 200),
        anomaly_std_threshold=3.0
    )
    
    # 3) Print the per-SKU metrics
    print("\n=== PER-SKU METRICS ===")
    print(df_sku_metrics)
    
    # 4) Show a sample of the augmented DataFrame with anomalies
    print("\n=== SAMPLE OF AUGMENTED DATA ===")
    print(df_pred_aug.head(10))
    
    # 5) Example: Check threshold-based "red flags" for each SKU
    print("\n=== RED FLAGS (Per SKU) ===")
    red_flags = {}
    for _, row in df_sku_metrics.iterrows():
        sku = row["SKU"]
        sku_flags = []
        
        if abs(row["Bias"]) > 5:
            sku_flags.append(f"High Bias: {row['Bias']:.2f}")
        if row["Anomaly_Rate"] > 0.05:
            sku_flags.append(f"Anomaly Rate > 5%: {row['Anomaly_Rate']:.2%}")
        if abs(row["Tracking_Signal"]) > 4:
            sku_flags.append(f"Tracking Signal out of +/-4: {row['Tracking_Signal']:.2f}")
        if (row["Direction_Accuracy"] is not None and 
            row["Direction_Accuracy"] < 0.6):
            sku_flags.append(f"Direction Accuracy < 60%: {row['Direction_Accuracy']:.2%}")
        if (row["PICP"] is not None and row["PICP"] < 0.7):
            sku_flags.append(f"PICP < 70%: {row['PICP']:.2%}")
        
        if sku_flags:
            red_flags[sku] = sku_flags
    
    if not red_flags:
        print("No red flags triggered.")
    else:
        for sku, flags in red_flags.items():
            print(f"\nSKU: {sku}")
            for f in flags:
                print(" -", f)