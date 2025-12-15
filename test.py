STATUS_MAP = {
    "Зареєстровано": "ACTIVE",

    "В стані припинення": "SUSPENDED",

    "Припинено": "TERMINATED",
    "Скасовано": "TERMINATED",
    "Архівний": "ARCHIVED",

    "Порушено справу про банкрутство": "BANKRUPTCY",
    "Порушено справу про банкрутство (санація)": "BANKRUPTCY",

    "Зареєстровано, свідоцтво про державну реєстрацію недійсне": "INVALID_REGISTRATION",
}

temp["status_std"] = (
    temp["Статус"]
    .map(STATUS_MAP)
    .fillna("OTHER")
)
