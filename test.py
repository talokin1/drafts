RE_UTILITIES = re.compile(
    r"""
    \b(
        комун|
        житлово[\s\-]?комунальн|
        ком\.?\s*послуг|
        електро|
        ел\.?\s*енерг|
        тепл(о|ов)|
        газ|
        вод(а|о)|
        оренд|
        паркомісц|
        відшк|
        рахунк|
        акт
    )\b
    """,
    re.IGNORECASE | re.VERBOSE
)


def detect_acquiring(row: pd.Series) -> pd.Series:
    pp_text = normalize_ua((row.get("PLATPURPOSE") or "").lower())
    cp_text = normalize_ua((row.get("CONTRAGENTASNAME") or "").lower())
    full_text = f"{pp_text} {cp_text}"

    if RE_UTILITIES.search(full_text):
        return pd.Series({
            "is_acquiring": False,
            "acq_reason": "utility_payment",
            "acq_score": 0,
        })


