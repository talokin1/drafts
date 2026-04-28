# Перезбереження моделі для мас-маркету
model_ms_to_save = LiabilitiesSmallModel(
    classifier=clf, # Твій навчений класифікатор (переконайся, що ім'я змінної правильне)
    regressor=reg,  # Твій навчений регресор
    cat_cols=cat_cols, 
    # СЕКРЕТ ТУТ: беремо список фічей прямо з "мізків" моделі
    feature_cols=clf.feature_name_, 
    classification_threshold=CLASSIFICATION_TRESHOLD 
)
joblib.dump(model_ms_to_save, r'C:\Projects\(DS-450) Corp_potential_income\scripts\models\pickle_models\Liabilities_SMALL.pkl')


# Перезбереження моделі для великого бізнесу
# (переконайся, що передаєш сюди правильні об'єкти clf_large/reg_large, якщо вони в тебе відрізняються)
model_large_to_save = LiabilitiesLargeModel(
    classifier=clf, 
    regressor=reg,  
    cat_cols=cat_cols,
    # СЕКРЕТ ТУТ: беремо список фічей прямо з "мізків" моделі
    feature_cols=clf.feature_name_, 
    classification_threshold=CLASSIFICATION_TRESHOLD 
)
joblib.dump(model_large_to_save, r'C:\Projects\(DS-450) Corp_potential_income\scripts\models\pickle_models\Liabilities_LARGE.pkl')