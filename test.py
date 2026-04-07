def calculate_interest_cover(row, limit_amount, interest_rate=0.20):
    if row['FIRM_TYPE'] == 'LARGE':
        hist_interest = row['A2250'] if not pd.isna(row['A2250']) else 0
    elif row['FIRM_TYPE'] in ['SMALL', 'MICRO']:
        hist_interest = row['A2270'] if not pd.isna(row['A2270']) else 0
    else:
        hist_interest = 0
    
    hist_interest = abs(hist_interest) 

    pot_interest = limit_amount * interest_rate

    total_interest = hist_interest + pot_interest

    if total_interest == 0:
        return np.inf
    
    ebitda = row['EBITDA'] if not pd.isna(row['EBITDA']) else 0
    
    return round(ebitda / total_interest, 2)


clients['IC_10'] = clients.apply(lambda row: calculate_interest_cover(row, limit_amount=10000), axis=1)

# І в cond для 10 млн:
(clients['IC_10'] >= 1.5) &