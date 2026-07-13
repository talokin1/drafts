import pandas as pd
from pathlib import Path


# 1. Шлях до нової нерозміченої вибірки
input_path = Path(
    r"C:\Projects\(DS-398) Acquiring\new_sample.xlsx"
)

# 2. Куди зберегти результат
output_path = input_path.with_name(
    f"{input_path.stem}_marked.xlsx"
)


# 3. Завантажуємо нову вибірку
df_new = pd.read_excel(input_path)


# 4. Перевіряємо наявність потрібних колонок
required_columns = [
    "PLATPURPOSE",
    "CONTRAGENTASNAME"
]

missing_columns = [
    col for col in required_columns
    if col not in df_new.columns
]

if missing_columns:
    raise ValueError(
        f"У файлі немає колонок: {missing_columns}"
    )


# 5. Застосовуємо правила до кожного рядка
detected = df_new.apply(
    detect_acquiring,
    axis=1
)


# 6. Додаємо результат до початкової вибірки
result = pd.concat(
    [
        df_new.reset_index(drop=True),
        detected.reset_index(drop=True)
    ],
    axis=1
)


# 7. Додаємо версію алгоритму
result["acq_rule_version"] = "regex_v1"


# 8. Виводимо коротку статистику
print("Кількість рядків:", len(result))

print(
    "Знайдено потенційного еквайрингу:",
    result["is_acquiring"].sum()
)

print(
    "Частка потенційного еквайрингу:",
    round(result["is_acquiring"].mean() * 100, 2),
    "%"
)

print("\nПричини спрацювання:")
print(
    result["acq_reason"]
    .value_counts(dropna=False)
)


# 9. Зберігаємо результат
result.to_excel(
    output_path,
    index=False
)

print("\nФайл збережено:")
print(output_path)