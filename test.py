pred = np.where(
    positive_proba >= MIN_PROBA_ASSETS,
    pred,
    0
)

pred = pd.Series(pred, index=X_part.index)

# segment-level calibration + caps
for seg in segment_part.unique():
    seg_mask = segment_part == seg

    factor = calibration_factors.get(seg, 1.0)
    cap = segment_caps.get(seg, np.inf)

    pred.loc[seg_mask] = pred.loc[seg_mask] * factor
    pred.loc[seg_mask] = np.clip(pred.loc[seg_mask], 0, cap)

final_predictions.loc[group_mask] = pred