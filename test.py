import numpy as np

# 1. Зсув всієї шкали NPS: 0–9 → 1–10
nps_data["score_adj"] = nps_data["Оцінка"] + 1

# 2. Введення no_info = 0 (якщо були NaN у сирих даних)
nps_data.loc[nps_data["Оцінка"].isna(), "score_adj"] = 0

# 3. Класифікація NPS (класична логіка)
def nps_class(score):
    if score == 0:
        return "no_info"
    elif score == 10:
        return "prom"
    elif score >= 8:
        return "neutr"
    else:
        return "detr"

nps_data["NPS"] = nps_data["score_adj"].apply(nps_class)

# 4. Контрольні перевірки
display(
    nps_data["score_adj"].value_counts().sort_index()
)

display(
    nps_data["NPS"].value_counts()
)
