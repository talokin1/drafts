# Список усіх колонок КВЕДів, які ми бачимо на скріншотах
kved_cols = [
    'KVED', 'KVED_DESCR',
    'KVED_2', 'KVED_2_DESCR',
    'KVED_3', 'KVED_3_DESCR',
    'KVED_4', 'KVED_4_DESCR',
    'KVED_5', 'KVED_5_DESCR'
]

# 1. Видаляємо старі колонки, які треба замінити
# errors='ignore' дозволяє уникнути помилки, якщо колонок вже немає
fin_ind = fin_ind.drop(columns=['FIRM_KVED', 'FIRM_KVEDNM'], errors='ignore')

# 2. Виконуємо злиття
# Беремо з ubki тільки ключ (IDENTIFYCODE) та потрібні колонки
fin_ind = fin_ind.merge(ubki[['IDENTIFYCODE'] + kved_cols], on='IDENTIFYCODE', how='left')