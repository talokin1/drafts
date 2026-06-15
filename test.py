Cards Current Month =
VAR CurrentMonth =
    MAX(PBI_Dim_Clients[Pilot Month])
RETURN
    CALCULATE(
        [Total Cards],
        FILTER(
            ALL(PBI_Dim_Clients[Pilot Month]),
            PBI_Dim_Clients[Pilot Month] = CurrentMonth
        )
    )



Cards Prev Month =
VAR CurrentMonth =
    MAX(PBI_Dim_Clients[Pilot Month])
VAR PrevMonth =
    CALCULATE(
        MAX(PBI_Dim_Clients[Pilot Month]),
        FILTER(
            ALL(PBI_Dim_Clients[Pilot Month]),
            PBI_Dim_Clients[Pilot Month] < CurrentMonth
        )
    )
RETURN
    CALCULATE(
        [Total Cards],
        FILTER(
            ALL(PBI_Dim_Clients[Pilot Month]),
            PBI_Dim_Clients[Pilot Month] = PrevMonth
        )
    )




Card MoM Diff =
[Cards Current Month] - [Cards Prev Month]






Card MoM Text =
IF(
    ISBLANK([Card MoM Diff]),
    BLANK(),
    "(" & FORMAT([Card MoM Diff], "+#,##0;-#,##0") & ")"
)