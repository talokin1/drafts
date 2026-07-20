counts = df["STRATIFY_TARGET"].value_counts()

df["STRATIFY_GROUP"] = df["STRATIFY_TARGET"].where(
    df["STRATIFY_TARGET"].map(counts) >= 5,
    "OTHER"
)

group_counts = df["STRATIFY_GROUP"].value_counts()
single_mask = df["STRATIFY_GROUP"].map(group_counts) < 2

df_main = df[~single_mask].copy()
df_single = df[single_mask].copy()

train, test = train_test_split(
    df_main,
    test_size=0.20,
    random_state=42,
    stratify=df_main["STRATIFY_GROUP"]
)

train = pd.concat([train, df_single], ignore_index=True)