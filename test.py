y = df[TARGET_COL].clip(lower=0)
X = df.drop(columns=[TARGET_COL])

y_cls = (y > 0).astype(int)
