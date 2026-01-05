import pandas as pd
import re
from pathlib import Path

DATA_DIR = Path(r"C:\Projects\Scorings\(DS-516) Corp Income\rr\source")


TYPES = [
]

MATURITY = [

]

def parse_reference_rate_xlsx(path: Path) -> pd.DataFrame:
    # 1️⃣ дата з назви файлу
    date = re.search(r"\d{4}\.\d{2}\.\d{2}", path.name).group()
    date = pd.to_datetime(date).strftime('%Y-%m-%d')

    # 2️⃣ читаємо Excel (беремо все як текст)
    df = pd.read_excel(path, header=None)

    # 3️⃣ витягуємо всі числа виду 12.34 або 12,34
    raw_numbers = re.findall(r"\d+[.,]\d+", df.to_string())
    raw_numbers = [float(x.replace(',', '.')) for x in raw_numbers]

    if len(raw_numbers) < 62:
        raise ValueError(f"{path.name}: знайдено менше 62 ставок")

    # 4️⃣ збираємо DataFrame
    temp = pd.DataFrame({
        'date': date,
        'type': TYPES,
        'maturity': MATURITY,
        'rr': raw_numbers[:62]
    })

    return temp


rr = []

for file in DATA_DIR.glob("Reference rate report *.xlsx"):
    print(f"Processing: {file.name}")
    rr.append(parse_reference_rate_xlsx(file))

rr = pd.concat(rr, ignore_index=True)


rr.to_csv(
    r"C:\Projects\Scorings\(DS-516) Corp Income\rr\rr_full_history.csv",
    index=False
)
