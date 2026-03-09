result = (
    directors
    .dropna(subset=["FIRSTCHIEF"])
    .groupby("IDENTIFYCODE")["FIRSTCHIEF"]
    .apply(list)
    .reset_index()
)

result