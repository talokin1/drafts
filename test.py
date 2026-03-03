import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================

PILOT_PATH = "Pilot_clients.xlsx"
MICRO_PATH = "Micro.xlsx"
SMALL_PATH = "Small.xlsx"
OUT_PATH   = "model_metrics_output.xlsx"

PILOT_ID_COL      = "IDENTIFYCO"      # перевір назву
PILOT_MONTH_COL   = "MONTH"
PILOT_PRIMARY_COL = "PRIMARY"

CRM_ID_COL     = "Ідентифікаційний номер"
CRM_RESULT_COL = "Результат дзвінка"

# --- статуси ---

MICRO_SUCCESS = {"Відкриття рахунку", "Рахунок відкрито", "Рахунок відкритою"}
MICRO_INPROG  = {"Зустріч на відділенні", "Необхідно подумати", "Передано на RM"}

SMALL_SUCCESS = {"Відкриття рахунку"}
SMALL_INPROG  = {"Клієнт зацікавлений",
                 "Консультацію проведено. Потрібен додатковий дзвінок"}

# =========================
# HELPERS
# =========================

def normalize_id(s):
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.replace(" ", "", regex=False)
    return s

def parse_primary(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.replace(",", ".").replace("%", "")
        try:
            val = float(x)
        except:
            return np.nan
    else:
        val = float(x)

    if val > 1:
        val /= 100
    return val

def classify(result, success_set, inprog_set):
    if pd.isna(result):
        return "fail"
    r = str(result).strip()
    if r in success_set:
        return "success"
    if r in inprog_set:
        return "in_progress"
    return "fail"

# =========================
# LOAD PILOT
# =========================

pilot = pd.read_excel(PILOT_PATH)

pilot["client_id"] = normalize_id(pilot[PILOT_ID_COL])
pilot["month"] = pilot[PILOT_MONTH_COL].astype(str)
pilot["primary"] = pilot[PILOT_PRIMARY_COL].apply(parse_primary)

pilot = pilot[["client_id", "month", "primary"]].drop_duplicates()

# =========================
# LOAD MICRO
# =========================

micro = pd.read_excel(MICRO_PATH)
micro["client_id"] = normalize_id(micro[CRM_ID_COL])
micro["segment"] = "Micro"
micro["status"] = micro[CRM_RESULT_COL].apply(
    lambda x: classify(x, MICRO_SUCCESS, MICRO_INPROG)
)
micro["taken"] = 1

# =========================
# LOAD SMALL
# =========================

small = pd.read_excel(SMALL_PATH)
small["client_id"] = normalize_id(small[CRM_ID_COL])
small["segment"] = "Small"
small["status"] = small[CRM_RESULT_COL].apply(
    lambda x: classify(x, SMALL_SUCCESS, SMALL_INPROG)
)
small["taken"] = 1

# =========================
# CRM MERGE
# =========================

crm = pd.concat([micro, small], ignore_index=True)

# якщо кілька записів по клієнту → беремо найсильніший статус
priority = {"success": 3, "in_progress": 2, "fail": 1}
crm["prio"] = crm["status"].map(priority)

crm = (
    crm.sort_values(["client_id", "segment", "prio"], ascending=False)
       .drop_duplicates(["client_id", "segment"])
       .drop(columns="prio")
)

# =========================
# MASTER TABLE
# =========================

master = pilot.merge(crm, on="client_id", how="left")

master["segment"] = master["segment"].fillna("No_Touch")
master["status"] = master["status"].fillna("not_taken")
master["taken"] = master["taken"].fillna(0)

master = master[master["segment"].isin(["Micro", "Small"])]

# =========================
# METRICS
# =========================

def calc_metrics(df):

    given = df["client_id"].nunique()
    taken = df[df["taken"] == 1]["client_id"].nunique()
    success = df[df["status"] == "success"]["client_id"].nunique()
    inprog = df[df["status"] == "in_progress"]["client_id"].nunique()
    fail = df[df["status"] == "fail"]["client_id"].nunique()
    not_taken = df[df["status"] == "not_taken"]["client_id"].nunique()

    avg_primary = df["primary"].mean()
    avg_succ = df[df["status"] == "success"]["primary"].mean()
    avg_fail = df[df["status"] == "fail"]["primary"].mean()

    return pd.Series({
        "GIVEN_CLIENTS": given,
        "ALL_TAKEN": taken,
        "SUCCESS": success,
        "IN_PROGRESS": inprog,
        "FAIL": fail,
        "NOT_TAKEN": not_taken,
        "TAKE_RATE": taken / given if given else 0,
        "CR_TAKEN": success / taken if taken else 0,
        "CR_GIVEN": success / given if given else 0,
        "AVG_PRIMARY": avg_primary,
        "AVG_PRIMARY_SUCCESS": avg_succ,
        "AVG_PRIMARY_FAIL": avg_fail,
        "LIFT": avg_succ - avg_fail if pd.notna(avg_succ) and pd.notna(avg_fail) else np.nan
    })

metrics = (
    master.groupby(["month", "segment"])
          .apply(calc_metrics)
          .reset_index()
)

# Pivot як на скріні
pivot = metrics.pivot(index="month", columns="segment")
pivot.columns = [f"{seg}_{metric}" for metric, seg in pivot.columns]
pivot = pivot.reset_index()

# =========================
# SAVE
# =========================

with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
    master.to_excel(writer, sheet_name="MASTER", index=False)
    metrics.to_excel(writer, sheet_name="METRICS_LONG", index=False)
    pivot.to_excel(writer, sheet_name="METRICS_PIVOT", index=False)

print("Saved:", OUT_PATH)