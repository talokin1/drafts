import pandas as pd
import re
import unicodedata

# 1. Твоя потужна функція нормалізації (адаптована)
_APOS_RE = re.compile(r"['’`’']")
_DASH_RE = re.compile(r"[-—−–]")

def norm_ua(s):
    if pd.isna(s) or s == "":
        return None
    
    s = str(s)
    # Нормалізація Unicode (щоб ї та i були правильними)
    s = unicodedata.normalize("NFKC", s).lower()
    
    # Видаляємо апострофи та замінюємо різні тире на дефіс
    s = _APOS_RE.sub("", s)
    s = _DASH_RE.sub("-", s)
    
    # Замінюємо будь-які лапки на пробіли, а потім зайві пробіли
    s = re.sub(r"[«»\"']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    
    return s

# 2. Застосовуємо нормалізацію до обох датафреймів
print("Нормалізація даних...")
final_df['FIRM_OPFNM_NORM'] = final_df['FIRM_OPFNM'].apply(norm_ua)
ubki['OPF_NORM'] = ubki['OPF'].apply(norm_ua)

# 3. Готуємо UBKI для мерджу
# Нам потрібні код, нормалізована назва (для логіки) та оригінальна назва (якщо хочете зберегти форматування джерела)
ubki_subset = ubki[['IDENTIFYCODE', 'OPF_CODE', 'OPF_NORM']].rename(columns={
    'OPF_CODE': 'UBKI_OPF_CODE',
    'OPF_NORM': 'UBKI_OPF_NAME'
})

# 4. Об'єднуємо (Merge)
final_df = final_df.merge(ubki_subset, on='IDENTIFYCODE', how='left')

# 5. Логіка заміни: Пріоритет UBKI -> Старі дані
# Заповнюємо коди
final_df['OPF_CODE'] = final_df['UBKI_OPF_CODE'].fillna(final_df['FIRM_OPFCD'])

# Заповнюємо назви (використовуємо нормалізовані версії для чистоти)
final_df['OPF_NAME'] = final_df['UBKI_OPF_NAME'].fillna(final_df['FIRM_OPFNM_NORM'])

# 6. Фінальна обробка: Робимо красиву велику літеру
# capitalize() робить першу букву великою, інші маленькими (Товариство з обмеженою...)
final_df['OPF_NAME'] = final_df['OPF_NAME'].str.capitalize()

# 7. Видаляємо старі та тимчасові колонки
cols_to_drop = ['FIRM_OPFCD', 'FIRM_OPFNM', 'FIRM_OPFNM_NORM', 'UBKI_OPF_CODE', 'UBKI_OPF_NAME']
# Перевіряємо, чи існують колонки перед видаленням, щоб не було помилки
final_df = final_df.drop(columns=[c for c in cols_to_drop if c in final_df.columns])

# Перевірка
print(final_df[['IDENTIFYCODE', 'OPF_CODE', 'OPF_NAME']].head())