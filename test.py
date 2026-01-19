probs = clf.predict_proba(X_val)[:,1]

best = None
for t in np.linspace(0.1, 0.9, 81):
    pred = np.zeros(len(X_val))
    m = probs > t
    if m.sum() > 0:
        pred[m] = np.expm1(reg.predict(X_val[m]))
    mae = mean_absolute_error(y_val, pred)
    if best is None or mae < best[0]:
        best = (mae, t, m.mean())
best
