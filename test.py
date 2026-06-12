pred_log_train = reg.predict(X_train_reg)
pred_log_val = reg.predict(X_val_reg)

pred_train = np.expm1(np.clip(pred_log_train, 0, None))
pred_val = np.expm1(np.clip(pred_log_val, 0, None))

print("TRAIN true max:", np.expm1(y_train_reg).max())
print("TRAIN pred max:", pred_train.max())

print("VAL true max:", np.expm1(y_val_reg).max())
print("VAL pred max:", pred_val.max())

print("\nTRAIN pred quantiles:")
print(pd.Series(pred_train).quantile([0.5, 0.75, 0.9, 0.95, 0.99, 0.999]))

print("\nVAL pred quantiles:")
print(pd.Series(pred_val).quantile([0.5, 0.75, 0.9, 0.95, 0.99, 0.999]))