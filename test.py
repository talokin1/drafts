def process_batch(df: pd.DataFrame) -> pd.DataFrame:
    detected = df.apply(detect_acquiring, axis=1)
    out = pd.concat([df.reset_index(drop=True), detected.reset_index(drop=True)], axis=1)
    return out
