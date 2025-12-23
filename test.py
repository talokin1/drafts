import numpy as np

nps_data["score_adj"] = np.where(
    nps_data["Оцінка"] == 0,
    0,                       # No_info
    nps_data["Оцінка"] + 1   # зсув 1–10
)

def nps_class(score):
    if score == 0:
        return "no_info"
    elif score >= 10:
        return "prom"
    elif score >= 8:
        return "neutr"
    else:
        return "detr"

nps_data["NPS"] = nps_data["score_adj"].apply(nps_class)

display(
    nps_data["score_adj"].value_counts().sort_index()
)

display(
    nps_data["NPS"].value_counts()
)
