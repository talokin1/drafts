import pandas as pd
import xlwings as xw

file_path = r"C:\Projects\(DS-644) Acquiring OTPay\Копія OTPay_Acquiring (002).xlsx"
sheet_name = "data"

# 1. Читаємо датафрейм як і раніше
df = pd.read_excel(file_path, engine="openpyxl", sheet_name=sheet_name)

# 2. Через Excel читаємо кольори
app = xw.App(visible=False)
app.display_alerts = False
app.screen_updating = False

try:
    wb = app.books.open(file_path)
    ws = wb.sheets[sheet_name]

    yellow_identifycodes = []

    # Визначимо останній рядок по колонці A
    last_row = ws.range("A" + str(ws.cells.last_cell.row)).end("up").row

    # Проходимо по рядках, починаючи з 2-го (1-й — header)
    for row in range(2, last_row + 1):
        # Перевіряємо колір у колонці C = FIRM_NAME
        color = ws.range(f"C{row}").color   # RGB tuple, наприклад (255, 255, 0)

        if color == (255, 255, 0):
            identifycode = ws.range(f"A{row}").value
            yellow_identifycodes.append(identifycode)

    wb.close()

finally:
    app.quit()

# 3. Нормалізуємо типи, щоб коректно відфільтрувати
df["IDENTIFYCODE"] = pd.to_numeric(df["IDENTIFYCODE"], errors="coerce")
yellow_identifycodes = pd.to_numeric(pd.Series(yellow_identifycodes), errors="coerce").dropna().tolist()

# 4. Видаляємо жовті рядки
df_clean = df[~df["IDENTIFYCODE"].isin(yellow_identifycodes)].copy()

# 5. За потреби окремо самі жовті
df_yellow = df[df["IDENTIFYCODE"].isin(yellow_identifycodes)].copy()

print("Жовтих рядків:", len(yellow_identifycodes))
print("Після видалення рядків:", len(df_clean))