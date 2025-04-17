import pandas as pd
import numpy as np
from scipy.stats import pearsonr, entropy

def jensen_shannon_distance(p, q):
    """
    p, q: discrete probability distributions (arrays summing to 1)
    JS distance = sqrt(0.5 * KL(p||m) + 0.5 * KL(q||m)), where m=(p+q)/2
    """
    p = np.asarray(p)
    q = np.asarray(q)
    m = 0.5 * (p + q)
    js_divergence = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
    return np.sqrt(js_divergence)

def compute_metrics_by_sku_and_actual_month(
    df_pred,
    data_anomaly_range=(10, 200),
    anomaly_std_threshold=3.0
):
    """
    Compute forecast metrics for EACH (SKU, Actual_Month).
    
    Grouping Key: (SKU, Actual_Month)
      -> Within each group, we have multiple "Prediction_Month" rows 
         (e.g., up to 12 months of forecast horizons).
    
    Columns expected in df_pred:
      - SKU
      - Actual_Month       (e.g., '2025-01' or some date-like string)
      - Actual_Value       (the known actual for that origin month)
      - Prediction_Month   (the future month being forecast)
      - Prediction_Value   (the forecast for that future month)
      - Prediction_Actual  (the true actual for that future month)
      (optional) Forecast_Lower, Forecast_Upper for intervals
    
    Parameters
    ----------
    df_pred : pd.DataFrame
        The expanded forecast data.
    data_anomaly_range : tuple
        (min_val, max_val) outside of which the "Actual_Value" is flagged as anomalous.
    anomaly_std_threshold : float
        Residuals beyond +/- threshold * std are flagged as anomalies.
    
    Returns
    -------
    df_pred_aug : pd.DataFrame
        The original df_pred plus columns:
          - Residual
          - Data_Anomaly_Flag
          - Anomaly_Flag
          - Within_Interval (if intervals exist)
    df_group_metrics : pd.DataFrame
        A table of metrics, one row per (SKU, Actual_Month), with columns like:
          [SKU, Actual_Month, Bias, Anomaly_Rate, PICP, Tracking_Signal, 
           Direction_Accuracy, Pearson_Correlation, R_squared, Jensen_Shannon_Distance, etc.]
    """
    # 1) Ensure required columns
    required_cols = ["SKU", "Actual_Month", "Actual_Value", 
                     "Prediction_Month", "Prediction_Value", "Prediction_Actual"]
    for c in required_cols:
        if c not in df_pred.columns:
            raise ValueError(f"Missing required column '{c}' in df_pred.")
    
    # 2) Calculate residual: forecast error for each row
    df_pred["Residual"] = df_pred["Prediction_Value"] - df_pred["Prediction_Actual"]
    
    # 3) Data anomaly: is the origin's Actual_Value out of a plausible range?
    (low_val, high_val) = data_anomaly_range
    df_pred["Data_Anomaly_Flag"] = df_pred["Actual_Value"].apply(
        lambda x: 1 if (x < low_val or x > high_val) else 0
    )
    
    # 4) PICP: only possible if Forecast_Lower & Forecast_Upper exist
    if "Forecast_Lower" in df_pred.columns and "Forecast_Upper" in df_pred.columns:
        df_pred["Within_Interval"] = df_pred.apply(
            lambda r: 1 if (r["Prediction_Actual"] >= r["Forecast_Lower"] and 
                            r["Prediction_Actual"] <= r["Forecast_Upper"]) else 0,
            axis=1
        )
    else:
        df_pred["Within_Interval"] = np.nan  # or 0 if you prefer
    
    # 5) Group by (SKU, Actual_Month)
    grouped = df_pred.groupby(["SKU", "Actual_Month"])
    
    results = []
    
    for (sku, origin_month), sub in grouped:
        # 'sub' is the set of rows for this SKU & this origin month,
        # each row is a different forecast horizon (Prediction_Month).
        
        # 5a) Compute residual-based anomaly
        res_std = sub["Residual"].std()
        if pd.isna(res_std) or res_std == 0:
            sub["Anomaly_Flag"] = 0
        else:
            upper_bound = anomaly_std_threshold * res_std
            lower_bound = -anomaly_std_threshold * res_std
            sub["Anomaly_Flag"] = sub["Residual"].apply(
                lambda x: 1 if (x < lower_bound or x > upper_bound) else 0
            )
        
        # Assign these anomaly flags back
        df_pred.loc[sub.index, "Anomaly_Flag"] = sub["Anomaly_Flag"]
        
        # 5b) Basic metrics
        bias = sub["Residual"].mean()                       # mean error
        anomaly_rate = sub["Anomaly_Flag"].mean()           # fraction of anomalies
        picp = sub["Within_Interval"].mean() if sub["Within_Interval"].notna().any() else np.nan
        
        # Tracking Signal = sum(residual) / (MAD * N)
        residual_sum = sub["Residual"].sum()
        mad = sub["Residual"].abs().mean()
        if (mad == 0) or (len(sub) == 0):
            tracking_signal = np.nan
        else:
            tracking_signal = residual_sum / (mad * len(sub))
        
        # 5c) Direction Accuracy for step=1 only, if your data has monthly steps:
        #    We'll parse the months as dates and see if Prediction_Month is exactly +1 from Actual_Month
        direction_checks = []
        
        # Attempt to convert to datetime; if not possible, fallback to NaN
        try:
            sub["AM_dt"] = pd.to_datetime(sub["Actual_Month"], format="%Y-%m", errors="coerce")
            sub["PM_dt"] = pd.to_datetime(sub["Prediction_Month"], format="%Y-%m", errors="coerce")
            
            for row in sub.itertuples(index=False):
                # If PM_dt is exactly 1 month after AM_dt
                if (row.PM_dt is not pd.NaT) and (row.AM_dt is not pd.NaT):
                    # Compare (year, month) logic
                    # e.g., if row.AM_dt = 2025-01, row.PM_dt = 2025-02 => step=1
                    am_year, am_month = row.AM_dt.year, row.AM_dt.month
                    pm_year, pm_month = row.PM_dt.year, row.PM_dt.month
                    # naive check:
                    if (pm_year == am_year and pm_month == am_month + 1) or \
                       (pm_year == am_year + 1 and (am_month == 12 and pm_month == 1)):
                        # direction
                        actual_change = row.Prediction_Actual - row.Actual_Value
                        forecast_change = row.Prediction_Value - row.Actual_Value
                        direction_checks.append(np.sign(actual_change) == np.sign(forecast_change))
        except:
            pass
        
        direction_accuracy = None
        if len(direction_checks) > 0:
            direction_accuracy = sum(direction_checks) / len(direction_checks)
        
        # 5d) Residual counts
        positive_res_count = (sub["Residual"] > 0).sum()
        negative_res_count = (sub["Residual"] < 0).sum()
        
        # 5e) Pearson correlation
        if len(sub) > 1:
            pearson_corr, _ = pearsonr(sub["Prediction_Value"], sub["Prediction_Actual"])
        else:
            pearson_corr = np.nan
        
        # 5f) R-squared
        y_true = sub["Prediction_Actual"]
        y_pred = sub["Prediction_Value"]
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        if ss_tot == 0:
            r_squared = np.nan
        else:
            r_squared = 1 - (ss_res / ss_tot)
        
        # 5g) Jensen-Shannon Distance between distributions of predicted vs actual
        if len(sub) > 1:
            all_min = min(y_true.min(), y_pred.min())
            all_max = max(y_true.max(), y_pred.max())
            if all_min == all_max:
                js_dist = 0.0
            else:
                bins = np.linspace(all_min, all_max, 10)
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
        
        results.append({
            "SKU": sku,
            "Actual_Month": origin_month,
            "Bias": bias,
            "Anomaly_Rate": anomaly_rate,
            "PICP": picp,
            "Tracking_Signal": tracking_signal,
            "Direction_Accuracy": direction_accuracy,
            "Positive_Residual_Count": positive_res_count,
            "Negative_Residual_Count": negative_res_count,
            "Pearson_Correlation": pearson_corr,
            "R_squared": r_squared,
            "Jensen_Shannon_Distance": js_dist
        })
    
    df_group_metrics = pd.DataFrame(results)
    return df_pred, df_group_metrics


# ---------------------------
# USAGE EXAMPLE (demo):
# ---------------------------
if __name__ == "__main__":
    # Suppose you have a CSV called "sample_data_generated.csv" 
    # with columns [SKU, Actual_Month, Actual_Value, Prediction_Month, Prediction_Value, Prediction_Actual, ...]
    
    # For demonstration, let's build a small example DataFrame by hand:
    data = [
        ["SKU_1", "2025-01", 120, "2025-02", 115, 110],
        ["SKU_1", "2025-01", 120, "2025-03", 118, 125],
        ["SKU_1", "2025-01", 120, "2025-04", 130, 128],
        ["SKU_1", "2025-02", 110, "2025-03", 112, 125],
        ["SKU_1", "2025-02", 110, "2025-04", 108, 128],
        
        ["SKU_2", "2025-01", 80,  "2025-02", 85,  78],
        ["SKU_2", "2025-01", 80,  "2025-03", 95,  88],
        ["SKU_2", "2025-01", 80,  "2025-04", 100, 102],
        # etc...
    ]
    df_demo = pd.DataFrame(data, columns=[
        "SKU", "Actual_Month", "Actual_Value", 
        "Prediction_Month", "Prediction_Value", "Prediction_Actual"
    ])
    
    # (Optional) add intervals if you have them
    # df_demo["Forecast_Lower"] = df_demo["Prediction_Value"] - 5
    # df_demo["Forecast_Upper"] = df_demo["Prediction_Value"] + 5
    
    # Now compute metrics per (SKU, Actual_Month)
    df_aug, df_grouped_metrics = compute_metrics_by_sku_and_actual_month(df_demo)
    
    # Print results
    print("\n=== AUGMENTED DF (first few rows) ===")
    print(df_aug.head(10))
    
    print("\n=== METRICS PER (SKU, Actual_Month) ===")
    print(df_grouped_metrics)