def compute_bias(actuals: pd.Series, preds: pd.Series) -> float:
    """
    Mean forecast error: mean(pred - actual).
    """
    residual = preds - actuals
    return residual.mean()


def compute_r_squared(actuals: pd.Series, preds: pd.Series) -> float:
    """
    R^2 = 1 - (SS_res / SS_tot), 
    where:
      SS_res = sum((actual - pred)^2),
      SS_tot = sum((actual - mean(actual))^2).
    """
    residual = preds - actuals
    ss_res = (residual ** 2).sum()
    ss_tot = ((actuals - actuals.mean()) ** 2).sum()
    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)


def compute_pearson_corr(actuals: pd.Series, preds: pd.Series) -> float:
    """
    Pearson correlation using Pandas' built-in .corr() method on Series.
    """
    return actuals.corr(preds)


def compute_residual_counts(actuals: pd.Series, preds: pd.Series) -> tuple:
    """
    Returns (count_of_positive_residuals, count_of_negative_residuals).
    Residual = pred - actual.
    """
    residual = preds - actuals
    pos_count = (residual > 0).sum()
    neg_count = (residual < 0).sum()
    return pos_count, neg_count


def compute_data_anomalies_shehd(actuals: pd.Series, preds: pd.Series) -> float:
    """
    Fraction of 'actuals' beyond mean ± 3*std (SHEHD style).
    """
    mean_a = actuals.mean()
    std_a = actuals.std()
    if pd.isna(std_a) or std_a == 0:
        return 0.0
    lower = mean_a - 3 * std_a
    upper = mean_a + 3 * std_a
    mask = (actuals < lower) | (actuals > upper)
    return mask.mean()  # fraction flagged


def compute_anomalies_shehd(actuals: pd.Series, preds: pd.Series) -> float:
    """
    Fraction of residuals beyond ±3*std(residual).
    """
    residual = preds - actuals
    std_r = residual.std()
    if pd.isna(std_r) or std_r == 0:
        return 0.0
    mask = residual.abs() > 3 * std_r
    return mask.mean()


def compute_tracking_signal(actuals: pd.Series, preds: pd.Series) -> float:
    """
    Tracking Signal = (Sum of residuals) / (MAD * N).
      residual = (pred - actual),
      MAD = mean(abs(residual)).
    """
    residual = preds - actuals
    sum_res = residual.sum()
    mad = residual.abs().mean()
    n = len(residual)
    if n == 0 or mad == 0:
        return np.nan
    return sum_res / (mad * n)


def compute_picp(actuals: pd.Series, preds: pd.Series) -> float:
    """
    Naive PICP with no explicit intervals:
      We'll pretend the forecast has ±10% intervals around preds,
      and see what fraction of actuals fall within [0.9*pred, 1.1*pred].
    """
    lower = preds * 0.9
    upper = preds * 1.1
    in_interval = (actuals >= lower) & (actuals <= upper)
    return in_interval.mean()


def compute_ause(actuals: pd.Series, preds: pd.Series) -> float:
    """
    Average Unsigned Scaled Error:
      AUSE = mean(|pred - actual| / (|actual| + epsilon))
    """
    epsilon = 1e-9
    ae = (preds - actuals).abs()
    scale = actuals.abs() + epsilon
    return (ae / scale).mean()


def compute_directional_accuracy(actuals: pd.Series, preds: pd.Series) -> float:
    """
    Fraction of times consecutive changes match in direction:
      sign(actual[i] - actual[i-1]) == sign(pred[i] - pred[i-1])
    Skips the first point, so requires at least 2 data points.
    """
    if len(actuals) < 2:
        return np.nan
    actual_diff = actuals.diff()  # change from row to row
    pred_diff = preds.diff()
    
    direction_mask = np.sign(actual_diff) == np.sign(pred_diff)
    # The first element of direction_mask is NaN (since diff(0) is NaN)
    # so skip it:
    return direction_mask.iloc[1:].mean()


def compute_data_drift_jsd(actuals: pd.Series, preds: pd.Series, bins=10) -> float:
    """
    Jensen-Shannon Distance between distributions of actuals and preds,
    using histogram-based approach with pd.cut().
    """
    if len(actuals) == 0 or len(preds) == 0:
        return np.nan
    
    min_val = min(actuals.min(), preds.min())
    max_val = max(actuals.max(), preds.max())
    if min_val == max_val:
        return 0.0  # all data is identical
    
    cats_a = pd.cut(actuals, bins=bins, include_lowest=True, labels=False)
    cats_p = pd.cut(preds, bins=bins, include_lowest=True, labels=False)
    
    counts_a = cats_a.value_counts().sort_index()
    counts_p = cats_p.value_counts().sort_index()
    
    if counts_a.sum() == 0 or counts_p.sum() == 0:
        return np.nan
    
    p = counts_a / counts_a.sum()
    q = counts_p / counts_p.sum()
    m = 0.5 * (p + q)
    
    p_arr = p.to_numpy()
    q_arr = q.to_numpy()
    m_arr = m.to_numpy()
    
    # Jensen-Shannon divergence
    js_divergence = 0.5 * entropy(p_arr, m_arr) + 0.5 * entropy(q_arr, m_arr)
    return np.sqrt(js_divergence)