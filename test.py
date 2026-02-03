# Замість простого map, явно вказуємо, що створюємо нову колонку
train_df[col + '_TE'] = train_df[col].map(smooth).astype(float)
val_df[col + '_TE'] = val_df[col].map(smooth).fillna(global_mean).astype(float)