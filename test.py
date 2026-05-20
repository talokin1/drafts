# ============================================================
# Money-based error columns
# ============================================================

eval_df["Actual Income"] = eval_df[FACT_COL]
eval_df["Predicted Income"] = eval_df[PRED_COL]

eval_df["Absolute Error Amount"] = (
    eval_df["Actual Income"] - eval_df["Predicted Income"]
).abs()

eval_df["Underestimation Amount"] = np.where(
    eval_df["Predicted Income"] < eval_df["Actual Income"],
    eval_df["Actual Income"] - eval_df["Predicted Income"],
    0
)

eval_df["Overestimation Amount"] = np.where(
    eval_df["Predicted Income"] > eval_df["Actual Income"],
    eval_df["Predicted Income"] - eval_df["Actual Income"],
    0
)

# Реальний дохід клієнтів, яких модель віднесла в нижчий Peter-клас
eval_df["Downgraded Actual Income"] = np.where(
    eval_df["Predicted Class Rank"] < eval_df["Actual Class Rank"],
    eval_df["Actual Income"],
    0
)

# Реальний дохід клієнтів, яких модель віднесла в вищий Peter-клас
eval_df["Upgraded Actual Income"] = np.where(
    eval_df["Predicted Class Rank"] > eval_df["Actual Class Rank"],
    eval_df["Actual Income"],
    0
)


def build_sum_matrix(df, value_col, labels):
    matrix = pd.pivot_table(
        df,
        values=value_col,
        index="Actual Peter Class",
        columns="Predicted Peter Class",
        aggfunc="sum",
        fill_value=0,
        observed=False
    )

    matrix = matrix.reindex(index=labels, columns=labels, fill_value=0)

    matrix["Grand Total"] = matrix.sum(axis=1)
    matrix.loc["Grand Total"] = matrix.sum(axis=0)

    matrix = matrix.round(0).astype(int)

    matrix = matrix.reset_index()
    matrix = matrix.rename(columns={"Actual Peter Class": "Actual Class"})

    return matrix



# ============================================================
# Money Confusion Matrices
# ============================================================

actual_income_matrix = build_sum_matrix(
    eval_df,
    value_col="Actual Income",
    labels=labels
)

predicted_income_matrix = build_sum_matrix(
    eval_df,
    value_col="Predicted Income",
    labels=labels
)

absolute_error_matrix = build_sum_matrix(
    eval_df,
    value_col="Absolute Error Amount",
    labels=labels
)

underestimation_matrix = build_sum_matrix(
    eval_df,
    value_col="Underestimation Amount",
    labels=labels
)

overestimation_matrix = build_sum_matrix(
    eval_df,
    value_col="Overestimation Amount",
    labels=labels
)

downgraded_actual_income_matrix = build_sum_matrix(
    eval_df,
    value_col="Downgraded Actual Income",
    labels=labels
)

upgraded_actual_income_matrix = build_sum_matrix(
    eval_df,
    value_col="Upgraded Actual Income",
    labels=labels
)



# ============================================================
# Money Summary Metrics
# ============================================================

total_actual_income = eval_df["Actual Income"].sum()
total_predicted_income = eval_df["Predicted Income"].sum()

total_abs_error = eval_df["Absolute Error Amount"].sum()
total_underestimation = eval_df["Underestimation Amount"].sum()
total_overestimation = eval_df["Overestimation Amount"].sum()

total_downgraded_actual_income = eval_df["Downgraded Actual Income"].sum()
total_upgraded_actual_income = eval_df["Upgraded Actual Income"].sum()

prediction_bias = total_predicted_income - total_actual_income
prediction_bias_pct = prediction_bias / total_actual_income if total_actual_income != 0 else np.nan

money_summary = pd.DataFrame({
    "Metric": [
        "Total Actual Income",
        "Total Predicted Income",
        "Prediction Bias",
        "Prediction Bias, %",
        "Total Absolute Error",
        "Total Underestimation Amount",
        "Total Overestimation Amount",
        "Downgraded Actual Income",
        "Upgraded Actual Income"
    ],
    "Value": [
        total_actual_income,
        total_predicted_income,
        prediction_bias,
        prediction_bias_pct,
        total_abs_error,
        total_underestimation,
        total_overestimation,
        total_downgraded_actual_income,
        total_upgraded_actual_income
    ],
    "Slide Format": [
        f"{total_actual_income:,.0f}".replace(",", " "),
        f"{total_predicted_income:,.0f}".replace(",", " "),
        f"{prediction_bias:,.0f}".replace(",", " "),
        f"{prediction_bias_pct * 100:.1f}%",
        f"{total_abs_error:,.0f}".replace(",", " "),
        f"{total_underestimation:,.0f}".replace(",", " "),
        f"{total_overestimation:,.0f}".replace(",", " "),
        f"{total_downgraded_actual_income:,.0f}".replace(",", " "),
        f"{total_upgraded_actual_income:,.0f}".replace(",", " ")
    ]
})

display(money_summary)









# ============================================================
# Write money matrices to report sheet
# ============================================================

money_start_row = insight_row + 7

money_tables = [
    ("Actual Income Matrix", actual_income_matrix),
    ("Predicted Income Matrix", predicted_income_matrix),
    ("Absolute Error Amount Matrix", absolute_error_matrix),
    ("Underestimation Amount Matrix", underestimation_matrix),
    ("Overestimation Amount Matrix", overestimation_matrix),
    ("Downgraded Actual Income Matrix", downgraded_actual_income_matrix),
    ("Upgraded Actual Income Matrix", upgraded_actual_income_matrix),
]

current_row = money_start_row

for title, table in money_tables:
    table_start_row = current_row

    current_row = write_df_to_sheet(
        ws,
        table,
        start_row=current_row,
        start_col=1,
        title=title
    )

    apply_table_style(
        ws,
        table_start_row + 1,
        current_row - 3,
        1,
        len(table.columns)
    )

    color_confusion_matrix(
        ws,
        start_row=table_start_row + 1,
        start_col=1,
        n_classes=5
    )

    # формат чисел як гроші без копійок
    for row_cells in ws.iter_rows(
        min_row=table_start_row + 2,
        max_row=current_row - 3,
        min_col=2,
        max_col=len(table.columns)
    ):
        for cell in row_cells:
            cell.number_format = '#,##0'

    current_row += 1


    