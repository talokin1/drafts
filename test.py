Cards Prev Month =
VAR CurrentMonth =
    SELECTEDVALUE(PBI_Fact_Income[Income_Date])
VAR PrevMonth =
    CALCULATE(
        MAX(PBI_Fact_Income[Income_Date]),
        FILTER(
            ALL(PBI_Fact_Income[Income_Date]),
            PBI_Fact_Income[Income_Date] < CurrentMonth
        )
    )
RETURN
    CALCULATE(
        [Total Cards],
        FILTER(
            ALL(PBI_Fact_Income[Income_Date]),
            PBI_Fact_Income[Income_Date] = PrevMonth
        )
    )





Card MoM Text =
IF(
    ISBLANK([Card MoM Diff]),
    BLANK(),
    "(" & FORMAT([Card MoM Diff], "+#,##0;-#,##0") & ")"
)