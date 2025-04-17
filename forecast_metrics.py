import numpy as np
import pandas as pd
from scipy.stats import pearsonr, entropy

def jensen_shannon_distance(p, q):
    """
    p, q: discrete probability distributions (arrays summing to 1)
    JS distance is sqrt(0.5 * KL(p||m) + 0.5 * KL(q||m)), where m = (p + q)/2
    """
    p = np.asarray(p)
    q = np.asarray(q)
    m = 0.5 * (p + q)
    js_divergence = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
    return np.sqrt(js_divergence)


def compute_metrics_by_sku(df_pred, data_anomaly_range=(10, 200), anomaly_std_threshold=3.0):
    """
    Compute forecast metrics on a *per-SKU* basis.
    
    Parameters
    ----------
    df_pred : pd.DataFrame
        Must have columns [SKU, Prediction_Value, Prediction_Actual, Forecast_Lower, 
                           Forecast_Upper, Actual_Value, (optionally more)].
    data_anomaly_range : tuple
        (min_value, max_value) outside of which 'Actual_Value' is flagged as anomalous.
    anomaly_std_threshold : float
        Residuals beyond +/- threshold * std are flagged as anomalies.

    Returns
    -------
    df_pred_aug : pd.DataFrame
        df_pred with additional columns: 'Residual', 'Anomaly_Flag', 'Data_Anomaly_Flag', 'Within_Interval'
    df_sku_metrics : pd.DataFrame
        A DataFrame of metrics, one row per SKU, columns = 
         [Bias, Anomaly_Rate, PICP, Tracking_Signal, Direction_Accuracy, 
          Positive_Residual_Count, Negative_Residual_Count, Pearson_Correlation, 
          R_squared, Jensen_Shannon_Distance]
    """
    
    # 1) Calculate the residual (global, for all SKUs)
    df_pred["Residual"] = df_pred["Prediction_Value"] - df_pred["Prediction_Actual"]
    
    # 2) Data anomaly flag
    low_range, high_range = data_anomaly_range
    df_pred["Data_Anomaly_Flag"] = df_pred["Actual_Value"].apply(
        lambda x: 1 if x < low_range or x > high_range else 0
    )
    
    # 3) Within interval for PICP
    df_pred["Within_Interval"] = df_pred.apply(
        lambda row: 1 if (row["Prediction_Actual"] >= row["Forecast_Lower"]) and
                     (row["Prediction_Actual"] <= row["Forecast_Upper"]) else 0,
        axis=1
    )
    
    # 4) We'll build a metrics table per SKU
    sku_list = df_pred["SKU"].unique()
    metrics_rows = []
    
    for sku in sku_list:
        sub = df_pred[df_pred["SKU"] == sku].copy()
        
        # Residual-based anomaly
        res_std = sub["Residual"].std()
        if pd.isna(res_std) or res_std == 0:
            sub["Anomaly_Flag"] = 0
        else:
            upper_bound = anomaly_std_threshold * res_std
            lower_bound = -anomaly_std_threshold * res_std
            sub["Anomaly_Flag"] = sub["Residual"].apply(
                lambda x: 1 if (x < lower_bound or x > upper_bound) else 0
            )
        
        # Persist these flags back to df_pred
        df_pred.loc[sub.index, "Anomaly_Flag"] = sub["Anomaly_Flag"]
        
        # --- Compute metrics for this SKU
        bias = sub["Residual"].mean()
        anomaly_rate = sub["Anomaly_Flag"].mean() if len(sub) else np.nan
        picp = sub["Within_Interval"].mean() if len(sub) else np.nan
        
        # Tracking Signal
        residual_sum = sub["Residual"].sum()
        mad = sub["Residual"].abs().mean()
        if mad == 0 or len(sub) == 0:
            tracking_signal = np.nan
        else:
            tracking_signal = residual_sum / (mad * len(sub))
        
        # Direction Accuracy (step_ahead=1)
        direction_checks = []
        for row in sub.itertuples(index=False):
            if row.Prediction_Month == row.Actual_Month + 1:
                actual_change = row.Prediction_Actual - row.Actual_Value
                forecast_change = row.Prediction_Value - row.Actual_Value
                direction_checks.append(np.sign(actual_change) == np.sign(forecast_change))
        direction_accuracy = None
        if len(direction_checks) > 0:
            direction_accuracy = sum(direction_checks) / len(direction_checks)
        
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