
excel_path = r"M:/Controlling/Data_Science_Projects/your_file.xlsx"

xls = pd.ExcelFile(excel_path)
months = xls.sheet_names


def process_month_sheet(df_month: pd.DataFrame, month: str) -> pd.DataFrame:
    detected = df_month.apply(detect_acquiring, axis=1)
    df_month = pd.concat([df_month, detected], axis=1)

    df_acq = df_month[df_month["is_acquiring"]].copy()
    if df_acq.empty:
        return pd.DataFrame()

    agg = (
        df_acq
        .groupby("CONTRAGENTBIDENTIFYCODE")
        .agg(
            SUMMAEQ=("SUMMAEQ", "sum"),
            NUM_TRX=("SUMMAEQ", "size"),
            PLATPURPOSE=("PLATPURPOSE", lambda x: x.dropna().iloc[0] if len(x.dropna()) else ""),
            CONTRAGENTASNAME=("CONTRAGENTASNAME", lambda x: x.dropna().iloc[0] if len(x.dropna()) else ""),
        )
        .reset_index()
        .rename(columns={"CONTRAGENTBIDENTIFYCODE": "IDENTIFYCODE"})
    )

    agg["IDENTIFYCODE"] = agg["IDENTIFYCODE"].astype(str)
    agg["MONTH"] = month

    return agg


all_months = []

for month in months:
    try:
        df_month = pd.read_excel(
            excel_path,
            sheet_name=month
        )

        agg_month = process_month_sheet(df_month, month)

        if not agg_month.empty:
            all_months.append(agg_month)

        print(f"[OK] {month}: {len(agg_month)} clients")

    except Exception as e:
        print(f"[ERROR] {month}: {e}")


acq_all = pd.concat(all_months, ignore_index=True)
acq_all


acq_all.to_excel(
    "M:/Controlling/Data_Science_Projects/acquiring_all_months.xlsx",
    index=False
)
