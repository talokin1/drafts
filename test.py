affluent_ap = average_precision_score((y >= 1).astype(int), oof_p_affluent)

mask = y >= 1
stage2_ap = average_precision_score((y.loc[mask] == 2).astype(int), p_hnwi_conditional[mask])

print("Stage 1 AP:", round(affluent_ap, 4), "| random:", round((y >= 1).mean(), 4))
print("Stage 2 AP:", round(stage2_ap, 4), "| random:", round((y.loc[mask] == 2).mean(), 4))