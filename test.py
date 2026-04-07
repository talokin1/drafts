def calculate_interest_cover(row, limit_amount, interest_rate=0.20):
    """
    Розрахунок Interest Cover з урахуванням історичних та потенційних відсотків.
    interest_rate (0.20) - індикативна ставка банку, узгодь її з ризиками.
    """
    if row.get('FIRM_TYPE') == 'LARGE':
        hist_interest = row.get('A2250') if not pd.isna(row.get('A2250')) else 0
    elif row.get('FIRM_TYPE') in ['SMALL', 'MICRO']:
        hist_interest = row.get('A2270') if not pd.isna(row.get('A2270')) else 0
    else:
        hist_interest = 0
        
    # Модуль обов'язковий, якщо в базі витрати йдуть з мінусом
    hist_interest = abs(hist_interest) 
    pot_interest = limit_amount * interest_rate
    total_interest = hist_interest + pot_interest
    
    if total_interest == 0:
        return np.inf
        
    ebitda = row.get('EBITDA') if not pd.isna(row.get('EBITDA')) else 0
    return round(ebitda / total_interest, 2)

clients["IC_10"] = clients.apply(lambda row: calculate_interest_cover(row, limit_amount=10000, interest_rate=0.20), axis=1)

(clients['IC_10'] >= 1.5)