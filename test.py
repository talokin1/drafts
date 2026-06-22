LIABS_FEATURES = [
    "CASH_CUR",
    "CASH_PREV",
    "ALR%_CUR",
    "CASH_DIF",
    "ALR%_PREV",
    "ROE%_CUR",
    "NB_EMP",
    "TAT_DIF",
    "ROE%_DIF",
    "ALR%_DIF",
    "ASSETS_DIF",
    "REVENUE_PREV",
    "INVENTORY_DIF",
    "ASSETS_PREV",
]

LIABS_FEATURE_WEIGHTS = {
    "CASH_CUR": 0.22,
    "CASH_PREV": 0.10,
    "ALR%_CUR": 0.09,
    "CASH_DIF": 0.09,
    "ALR%_PREV": 0.07,
    "ROE%_CUR": 0.06,
    "NB_EMP": 0.06,
    "TAT_DIF": 0.06,
    "ROE%_DIF": 0.05,
    "ALR%_DIF": 0.05,
    "ASSETS_DIF": 0.05,
    "REVENUE_PREV": 0.04,
    "INVENTORY_DIF": 0.03,
    "ASSETS_PREV": 0.03,
}




ASSETS_FEATURES = [
    "INVENTORY_CUR",
    "CURRENT_ASSETS_CUR",
    "ASSETS_CUR",
    "CR%_DIF",
    "DSO_PREV",
    "OPM%_DIF",
    "FIXED_ASSETS_PREV",
    "PAYABLES_DIF",
    "LIQUID_ASSETS_DIF",
    "LTFR_DIF",
]

ASSETS_FEATURE_WEIGHTS = {
    "INVENTORY_CUR": 0.18,
    "CURRENT_ASSETS_CUR": 0.16,
    "ASSETS_CUR": 0.14,
    "CR%_DIF": 0.11,
    "DSO_PREV": 0.10,
    "OPM%_DIF": 0.09,
    "FIXED_ASSETS_PREV": 0.07,
    "PAYABLES_DIF": 0.06,
    "LIQUID_ASSETS_DIF": 0.05,
    "LTFR_DIF": 0.04,
}



FX_FEATURES = [
    "IMPORT_USD",
    "A1165",
    "EXPORT_USD",
    "DIO_CUR",
    "A2160",
    "A2120",
    "TAT_CUR",
    "A2285",
    "B2120",
    "DIO_PREV",
]

FX_FEATURE_WEIGHTS = {
    "IMPORT_USD": 0.30,
    "A1165": 0.15,
    "EXPORT_USD": 0.14,
    "DIO_CUR": 0.09,
    "A2160": 0.08,
    "A2120": 0.08,
    "TAT_CUR": 0.05,
    "A2285": 0.05,
    "B2120": 0.03,
    "DIO_PREV": 0.03,
}