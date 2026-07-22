TARGET_MAP = {"MASS": 0, "PREM": 1, "HNWI": 2}
TARGET_NAMES = np.array(["MASS", "PREM", "HNWI"])

data = client_df.reset_index(drop=True).copy()
groups = data["group_id"].astype(str)
y = data["SEGMENT"].map(TARGET_MAP).astype(int)
X = data.drop(columns=["MOBILEPHONE", "group_id", "SEGMENT", "CONTRAGENTID"], errors="ignore").replace([np.inf, -np.inf], np.nan)

cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

for col in cat_cols:
    X[col] = X[col].astype("string").fillna("Missing").astype(str)