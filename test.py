# 0-10k: крок 2.5k | 10k-25k: крок 5k | 25k-100k: крок 10k
bins = [-1, 2500, 5000, 7500, 10000, 15000, 20000, 25000]
bins += list(range(30000, 100001, 10000))
bins += [1000000, np.inf]
bins = sorted(set(bins))

labels = []
for i in range(len(bins)-1):
    low = bins[i]
    high = bins[i+1]
    
    if high == np.inf:
        labels.append(f"{int(low/1000)}k+")
    elif low == -1:
        labels.append(f"0-{high/1000:g}k")
    else:
        labels.append(f"{low/1000:g}k-{high/1000:g}k")

df["Income_Bucket_TRUE"] = pd.cut(df[TRUE_COL], bins=bins, labels=labels)
df["Income_Bucket_PRED"] = pd.cut(df[PRED_COL], bins=bins, labels=labels)


