import pandas as pd
import re
import unicodedata
import pymorphy3

# --- 1. ВАШІ ФУНКЦІЇ НОРМАЛІЗАЦІЇ ---
_APOS_RE = re.compile(r"['’`’']")
_DASH_RE = re.compile(r"[-—−–]")

def norm_ua(s):
    if pd.isna(s) or s == "":
        return None
    s = str(s)
    s = unicodedata.normalize("NFKC", s).lower()
    s = _APOS_RE.sub("", s)
    s = _DASH_RE.sub("-", s)
    s = re.sub(r"[«»\"']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --- 2. ЗАВАНТАЖЕННЯ ДОВІДНИКА (З вашого коду) ---
# Вкажіть правильний шлях до файлу, як було у вас
opf_path = r"M:\Controlling\Data_Science_Projects\Financial_indicators_tool\OPFCD_OPFNM.xls"
# Якщо файл не доступний зараз, закоментуйте цей блок
try:
    opf_df = pd.read_excel(opf_path)
    
    opf_ref = (
        opf_df[["FIRM_OPFCD", "FIRM_OPFNM"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"FIRM_OPFCD": "OPF_CODE", "FIRM_OPFNM": "OPF_NAME"})
    )
    
    # Нормалізуємо довідник для пошуку
    opf_ref["OPF_NAME_NORM"] = opf_ref["OPF_NAME"].map(norm_ua)
    opf_ref["OPF_CODE_INT"] = opf_ref["OPF_CODE"].astype(int) # Для точного збігу

    # Сортуємо: найдовші назви зверху (щоб "Товариство..." знайшлося раніше ніж "Товариство")
    opf_ref_exact = opf_ref.sort_values("OPF_NAME_NORM", key=lambda s: s.str.len(), ascending=False).reset_index(drop=True)

except Exception as e:
    print(f"Увага: Довідник не завантажено. {e}")
    opf_ref_exact = pd.DataFrame()

# --- 3. ФУНКЦІЯ EXTRACT_OPF (Відновлена) ---
# !ВАЖЛИВО: Переконайтесь, що ALL_MAP_MARKERS визначено у вас в коді
# Якщо немає, створіть пустий dict або додайте ваші маркери
if 'ALL_MAP_MARKERS' not in locals():
    ALL_MAP_MARKERS = [] # Вставте сюди ваші словники маркерів, якщо вони є

def extract_opf(full_name_norm):
    if not full_name_norm or opf_ref_exact.empty:
        return None, None, "UNKNOWN"

    # LEVEL 1: Exact match (starts with)
    for _, r in opf_ref_exact.iterrows():
        if full_name_norm.startswith(r["OPF_NAME_NORM"]):
            return r["OPF_CODE"], r["OPF_NAME"], "EXACT"

    # LEVEL 2: Marker based (якщо у вас є маркери)
    # Це логіка з вашого скріншоту
    for marker_group in ALL_MAP_MARKERS:
        for opf_code, patterns in marker_group.items():
            for pattern in patterns:
                if re.search(pattern, full_name_norm):
                    # Знаходимо назву по коду
                    name_row = opf_ref[opf_ref["OPF_CODE_INT"] == opf_code]
                    if not name_row.empty:
                        return opf_code, name_row.iloc[0]["OPF_NAME"], "MARKER"

    return None, None, "UNKNOWN"

# --- 4. ОБРОБКА FINAL_DF ---

print("Нормалізація вхідних даних...")
# Нормалізуємо колонку, де лежить назва ОПФ або повна назва фірми
# Використовуємо FIRM_OPFNM або FIRM_NAME для пошуку
final_df['TEMP_NORM_NAME'] = final_df['FIRM_OPFNM'].fillna(final_df['FIRM_NAME']).apply(norm_ua)

print("Запуск extract_opf (це може зайняти час)...")
# Застосовуємо вашу функцію до кожного рядка
extracted_data = final_df['TEMP_NORM_NAME'].apply(lambda x: pd.Series(extract_opf(x), index=['EXT_CODE', 'EXT_NAME', 'EXT_SOURCE']))

# Додаємо результати extraction в датафрейм
final_df = pd.concat([final_df, extracted_data], axis=1)

# --- 5. ОБРОБКА UBKI ТА MERGE ---
ubki['OPF_NORM'] = ubki['OPF'].apply(norm_ua)
ubki_subset = ubki[['IDENTIFYCODE', 'OPF_CODE', 'OPF_NORM']].rename(columns={
    'OPF_CODE': 'UBKI_OPF_CODE',
    'OPF_NORM': 'UBKI_OPF_NAME' # Тут, можливо, треба original name, якщо хочете красиво
})

final_df = final_df.merge(ubki_subset, on='IDENTIFYCODE', how='left')

# --- 6. ФІНАЛЬНА КОНСОЛІДАЦІЯ (PRIORITY LOGIC) ---
# Логіка:
# 1. Якщо є в UBKI -> беремо UBKI
# 2. Якщо немає в UBKI, але extract_opf знайшов ("EXACT" або "MARKER") -> беремо EXTRACTED
# 3. Якщо нічого немає -> беремо старе FIRM_OPFCD

# --- КОД (OPF_CODE) ---
final_df['OPF_CODE'] = final_df['UBKI_OPF_CODE'] # Пріоритет 1
final_df['OPF_CODE'] = final_df['OPF_CODE'].fillna(final_df['EXT_CODE']) # Пріоритет 2
final_df['OPF_CODE'] = final_df['OPF_CODE'].fillna(final_df['FIRM_OPFCD']) # Пріоритет 3

# --- НАЗВА (OPF_NAME) ---
# Для UBKI ми маємо тільки нормалізовану назву або код. 
# Якщо треба гарна назва, краще брати з довідника по коду UBKI, або форматувати текст.
final_df['OPF_NAME'] = final_df['UBKI_OPF_NAME']
final_df['OPF_NAME'] = final_df['OPF_NAME'].fillna(final_df['EXT_NAME'])
final_df['OPF_NAME'] = final_df['OPF_NAME'].fillna(final_df['FIRM_OPFNM'])

# --- 7. КОСМЕТИКА (Capitalize) ---
final_df['OPF_NAME'] = final_df['OPF_NAME'].astype(str).str.capitalize()

# Чистимо сміття
cols_to_drop = ['TEMP_NORM_NAME', 'EXT_CODE', 'EXT_NAME', 'EXT_SOURCE', 
                'UBKI_OPF_CODE', 'UBKI_OPF_NAME', 'FIRM_OPFCD', 'FIRM_OPFNM']
final_df = final_df.drop(columns=[c for c in cols_to_drop if c in final_df.columns])

print("Готово! Перші 5 рядків:")
print(final_df[['IDENTIFYCODE', 'OPF_CODE', 'OPF_NAME']].head())