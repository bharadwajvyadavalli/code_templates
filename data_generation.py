import numpy as np
import pandas as pd

def _month_str_to_int(m_str):
    # Convert "M1" -> 1, "M12" -> 12, etc.
    return int(m_str.replace("M", ""))

def generate_baseline_data(num_skus=10, n_months=24, seed=42):
    """
    Generates baseline actual data for `num_skus` SKUs, each with `n_months` labeled as M1..M24.
    Then creates naive forecasts for each (SKU, Actual_Month) up to 12 months ahead (if available).
    
    Returns
    -------
    df_pred : pd.DataFrame
        Columns:
          SKU, Actual_Month, Actual_Value, Prediction_Month, Prediction_Value, Prediction_Actual
        (No anomalies injected yet; still floats at this stage.)
    """
    np.random.seed(seed)
    
    skus = [f"SKU_{i+1}" for i in range(num_skus)]
    actual_rows = []
    
    # 1) Generate actual data
    for sku in skus:
        baseline = 100 + np.random.normal(0, 10)
        trend = np.random.normal(0, 2)
        
        for m in range(1, n_months+1):
            # Synthetic actual: baseline + trend*m + noise
            val = baseline + (trend * m) + np.random.normal(0, 5)
            val = max(val, 0)  # ensure non-negative
            actual_rows.append((sku, f"M{m}", val))
    
    df_actual = pd.DataFrame(actual_rows, columns=["SKU", "Actual_Month", "Actual_Value"])
    
    # 2) Build naive forecasts
    forecast_rows = []
    for sku in skus:
        sku_data = df_actual[df_actual["SKU"] == sku].copy()
        sku_data["month_int"] = sku_data["Actual_Month"].apply(_month_str_to_int)
        
        # quick lookup: month_int -> actual_value
        lookup = dict(zip(sku_data["month_int"], sku_data["Actual_Value"]))
        
        # For each actual month, forecast up to 12 future months
        for row in sku_data.itertuples(index=False):
            current_m_int = row.month_int
            current_val = row.Actual_Value
            for horizon in range(1, 13):
                future_m_int = current_m_int + horizon
                if future_m_int <= n_months:
                    future_val = lookup[future_m_int]
                    # naive forecast = future_val + small noise
                    pred_val = future_val + np.random.normal(0, 5)
                    forecast_rows.append([
                        sku,
                        f"M{current_m_int}",
                        current_val,
                        f"M{future_m_int}",
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
    bias_month_range=(10, 12),
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
    3) Systematic bias: for the specified SKU(s) and origin month range,
       add bias_shift to 'Prediction_Value'.
    
    Returns
    -------
    df_pred_aug : DataFrame
        Same columns, but now containing forced anomalies and bias spikes (still floats at this point).
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
    
    # 3) Systematic bias for certain SKUs and origin month range
    start_m, end_m = bias_month_range
    
    def month_int(m_str):
        return int(m_str.replace("M", ""))
    
    for idx, row in df.iterrows():
        if row["SKU"] in bias_sku_list:
            origin_m = month_int(row["Actual_Month"])
            if start_m <= origin_m <= end_m:
                df.at[idx, "Prediction_Value"] += bias_shift
    
    return df


def main():
    # 1) Generate baseline data for 10 SKUs, 24 months
    df_pred = generate_baseline_data(num_skus=10, n_months=24, seed=42)
    
    # 2) Inject anomalies & bias
    df_pred_aug = inject_anomalies_and_bias(
        df_pred,
        data_anomaly_prob=0.08,       # 8% data anomalies
        data_anomaly_factor=5.0,
        residual_anomaly_prob=0.10,   # 10% big forecast errors
        residual_anomaly_factor=30.0,
        bias_sku_list=["SKU_1", "SKU_2"],
        bias_month_range=(10, 12),
        bias_shift=25.0,
        seed=999
    )
    
    # 3) Convert all numeric columns to integer (round first).
    #    This includes actual and predicted values.
    df_pred_aug["Actual_Value"] = np.round(df_pred_aug["Actual_Value"]).astype(int)
    df_pred_aug["Prediction_Value"] = np.round(df_pred_aug["Prediction_Value"]).astype(int)
    df_pred_aug["Prediction_Actual"] = np.round(df_pred_aug["Prediction_Actual"]).astype(int)
    
    print("\n=== FINAL df_pred_aug (INT VALUES, with forced anomalies & bias) ===")
    print(df_pred_aug.head(20))
    
    # Optionally save to CSV:
    # df_pred_aug.to_csv("sample_data_with_anomalies_int.csv", index=False)


if __name__ == "__main__":
    main()