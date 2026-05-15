import pandas as pd
import numpy as np

# ============================================================
# 1. Список КВЕДів з лівої колонки на фото
# ============================================================

kved_codes_raw = [
    2110, 2120,
    2611, 2612, 2620, 2630, 2640, 2651, 2652, 2660, 2670, 2680,
    3250,
    3831, 3832,
    4623, 4624, 4631, 4632, 4633, 4634, 4636, 4637, 4638, 4639,
    4646, 4675,
    4711, 4721, 4722, 4723, 4724, 4729,
    4773, 4774, 4781,
    5210, 5221, 5222, 5224, 5229,
    5310, 5320,
    5821, 5829,
    6110, 6120, 6130, 6190,
    6201, 6202, 6203, 6209,
    6311, 6312, 6399,
    8610, 8621, 8622, 8623, 8690,
    9810, 9820, 9900
]


# ============================================================
# 2. Функція для приведення КВЕДу до формату XX.XX
# ============================================================

def normalize_kved(value):
    """
    Приводить КВЕД до формату XX.XX.

    Приклади:
    2110        -> 21.10
    "2110"      -> 21.10
    "21.10"     -> 21.10
    "C21.10"    -> 21.10
    "C21.10.0"  -> 21.10
    """

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    digits = "".join([ch for ch in value if ch.isdigit()])

    if len(digits) < 4:
        return np.nan

    digits = digits[:4]

    return f"{digits[:2]}.{digits[2:]}"


# ============================================================
# 3. Функція для нормалізації OPF_CODE
# ============================================================

def normalize_opf_code(value):
    """
    Приводить OPF_CODE до 4-значного string-формату.

    Приклади:
    1400      -> "1400"
    "1400"    -> "1400"
    140.0     -> "1400"
    """

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    # Якщо значення прийшло як 1400.0
    if value.endswith(".0"):
        value = value[:-2]

    digits = "".join([ch for ch in value if ch.isdigit()])

    if len(digits) == 0:
        return np.nan

    return digits.zfill(4)


# ============================================================
# 4. Нормалізація списку потрібних КВЕДів
# ============================================================

target_kveds = [normalize_kved(code) for code in kved_codes_raw]
target_kveds = sorted(list(set(target_kveds)))

print("Кількість КВЕДів у фільтрі:", len(target_kveds))
print(target_kveds)


# ============================================================
# 5. OPF_CODE, які треба виключити
# ============================================================

exclude_opf_codes = [
    "1400",  # Державне підприємство
    "1450",  # Казенне підприємство
    "1500",  # Комунальне підприємство
]

exclude_opf_codes = set(exclude_opf_codes)


# ============================================================
# 6. Підготовка таблиці
# ============================================================

fin_ind = fin_ind.copy()

# Нормалізуємо КВЕД
fin_ind["KVED_norm"] = fin_ind["KVED"].apply(normalize_kved)

# Нормалізуємо OPF_CODE
fin_ind["OPF_CODE_norm"] = fin_ind["OPF_CODE"].apply(normalize_opf_code)

# Приводимо числові поля до числового типу
numeric_cols = ["NET_PROFIT_CUR", "REVENUE_CUR", "NB_EMPL"]

for col in numeric_cols:
    fin_ind[col] = pd.to_numeric(fin_ind[col], errors="coerce")


# ============================================================
# 7. Маска для державних та комунальних підприємств
# ============================================================

is_state_or_municipal = fin_ind["OPF_CODE_norm"].isin(exclude_opf_codes)


# ============================================================
# 8. Формування фінальної вибірки
# ============================================================

sample = fin_ind[
    (fin_ind["NET_PROFIT_CUR"] > 0) &
    (fin_ind["REVENUE_CUR"] > 70_000) &
    (fin_ind["NB_EMPL"] > 30) &
    (fin_ind["NB_EMPL"] < 300) &
    (fin_ind["KVED_norm"].isin(target_kveds)) &
    (~is_state_or_municipal)
].copy()


# ============================================================
# 9. Результати
# ============================================================

print("Початкова кількість рядків:", len(fin_ind))
print("Кількість рядків з державними/комунальними OPF_CODE:", is_state_or_municipal.sum())
print("Кількість рядків після фільтра:", len(sample))

if "IDENTIFYCODE" in sample.columns:
    print("Кількість унікальних клієнтів:", sample["IDENTIFYCODE"].nunique())

display(sample.head())


# ============================================================
# 10. Перевірка OPF_CODE, які були виключені
# ============================================================

excluded_opf_distribution = (
    fin_ind[is_state_or_municipal]
    .groupby(["OPF_CODE_norm", "OPF_NAME"], dropna=False)
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

display(excluded_opf_distribution)


# ============================================================
# 11. Перевірка OPF_CODE, які залишились після фільтра
# ============================================================

opf_distribution_after_filter = (
    sample
    .groupby(["OPF_CODE_norm", "OPF_NAME"], dropna=False)
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

display(opf_distribution_after_filter)


# ============================================================
# 12. Розподіл по КВЕДах після фільтра
# ============================================================

kved_distribution = (
    sample["KVED_norm"]
    .value_counts()
    .reset_index()
)

kved_distribution.columns = ["KVED", "count"]

display(kved_distribution)


# ============================================================
# 13. Перевірка, які КВЕДи зі списку реально знайшлись
# ============================================================

found_kveds = sorted(sample["KVED_norm"].dropna().unique())
not_found_kveds = sorted(set(target_kveds) - set(found_kveds))

print("КВЕДи, які знайшлись у вибірці:")
print(found_kveds)

print("\nКВЕДи зі списку, яких немає у вибірці:")
print(not_found_kveds)


# ============================================================
# 14. Збереження результатів
# ============================================================

sample.to_excel("filtered_clients_final.xlsx", index=False)
kved_distribution.to_excel("kved_distribution.xlsx", index=False)
opf_distribution_after_filter.to_excel("opf_distribution_after_filter.xlsx", index=False)
excluded_opf_distribution.to_excel("excluded_opf_distribution.xlsx", index=False)

print("Файли збережено:")
print("filtered_clients_final.xlsx")
print("kved_distribution.xlsx")
print("opf_distribution_after_filter.xlsx")
print("excluded_opf_distribution.xlsx")


exclude_opf_codes = {
    "1400",  # Державне підприємство
    "1450",  # Казенне підприємство
    "1500",  # Комунальне підприємство
}

is_state_or_municipal = fin_ind["OPF_CODE_norm"].isin(exclude_opf_codes)

sample = fin_ind[
    ...
    (~is_state_or_municipal)
].copy()