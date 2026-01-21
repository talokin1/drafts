BUSINESS_MAP = {
    'BUSINESS_B1': 'Зарплатний проєкт компанії клієнта в ОТП',
    'BUSINESS_B2': 'Бізнес клієнта (компанія / ФОП) в ОТП',
    'BUSINESS_B3': 'Виплата зарплати на рахунки ОТП (останні 6 міс.)',
    'BUSINESS_B4': 'Виплата іншого доходу (дивіденди, роялті тощо)'
}


PORTFOLIO_MAP = {
    'PORTFOLIO_P1': 'Строкові депозити (кожні 500k UAH)',
    'PORTFOLIO_P2': 'Депозити в UAH / FCY (від 1kk eq)',
    'PORTFOLIO_P3': 'Інвестиційний портфель (кожен 1kk UAH eq)',
    'PORTFOLIO_P4': 'Цінні папери (ISIN ≥ 2)'
}

DAILY_MAP = {
    'DAILY_D1':  'Валютообмін у застосунку (3 міс.)',
    'DAILY_D2':  'ОВДП у застосунку (3 міс.)',
    'DAILY_D3':  'Перекази у застосунку (30 днів)',
    'DAILY_D4':  'Страхування від шахрайства',
    'DAILY_D5':  'SWIFT-платежі (6 міс., ≥ 400k UAH eq)',
    'DAILY_D6':  'Розрахунок карткою (50–100k / 100k / 250k)',
    'DAILY_D9':  'Розрахунок за кордоном (≥ 30k, 3 міс.)',
    'DAILY_D10': 'Готелі / авіаквитки / оренда авто (≥ 30k, 3 міс.)',
    'DAILY_D11': 'Аеропортові сервіси (3 міс.)'
}


def decode_drivers(driver_list, mapping):
    if not driver_list:
        return []
    return [mapping.get(d, d) for d in driver_list]

cmp['TOP_BUSINESS_DRIVERS_TXT'] = cmp['TOP_BUSINESS_DRIVERS'].apply(
    lambda x: decode_drivers(x, BUSINESS_MAP)
)

cmp['TOP_PORTFOLIO_DRIVERS_TXT'] = cmp['TOP_PORTFOLIO_DRIVERS'].apply(
    lambda x: decode_drivers(x, PORTFOLIO_MAP)
)

cmp['TOP_DAILY_DRIVERS_TXT'] = cmp['TOP_DAILY_DRIVERS'].apply(
    lambda x: decode_drivers(x, DAILY_MAP)
)


def explain_growth_human(row):
    if row['MAIN_GROWTH_DRIVER'] == 'BUSINESS':
        return f"Business: {row['TOP_BUSINESS_DRIVERS_TXT']}"
    if row['MAIN_GROWTH_DRIVER'] == 'PORTFOLIO':
        return f"Portfolio: {row['TOP_PORTFOLIO_DRIVERS_TXT']}"
    return f"Daily banking: {row['TOP_DAILY_DRIVERS_TXT']}"
cmp['GROWTH_EXPLANATION_HUMAN'] = cmp.apply(explain_growth_human, axis=1)


final_summary = (
    cmp
    .assign(
        TOTAL_SCORE_CHANGE=lambda x:
            x['TOTAL_SCORE_NOV'].astype(int).astype(str)
            + ' → '
            + x['TOTAL_SCORE_MAR'].astype(int).astype(str)
    )
    [[
        'CONTRAGENTID',
        'TOTAL_SCORE_CHANGE',
        'DELTA_TOTAL_SCORE',
        'MAIN_GROWTH_DRIVER',
        'GROWTH_EXPLANATION_HUMAN'
    ]]
    .sort_values('DELTA_TOTAL_SCORE', ascending=False)
    .reset_index(drop=True)
)
