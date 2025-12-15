temp = temp.rename(columns={
    "Актуально на": "ubki_actual_date",
    "Останні зміни": "registry_last_change_date"
})

temp["ubki_update_lag_days"] = (
    temp["ubki_actual_date"] - temp["registry_last_change_date"]
).dt.days

today_ref = temp["ubki_actual_date"].max()

temp["days_since_registry_change"] = (
    today_ref - temp["registry_last_change_date"]
).dt.days

today_ref = temp["ubki_actual_date"].max()

temp["days_since_registry_change"] = (
    today_ref - temp["registry_last_change_date"]
).dt.days
