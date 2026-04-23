pattern = r"""
(
пдфо|
єсв|
єсф|
єдиний\s*соц|
військ\w*\s*зб\w*
)
"""

result['IS_TAX'] = result['PLATPURPOSE'].str.lower().str.contains(
    pattern,
    regex=True,
    na=False
)