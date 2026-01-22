import pymorphy3
morph = pymorphy3.MorphAnalyzer(lang="uk")

import re

def lemmatize_ua(text: str) -> list[str]:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = re.findall(r"[а-яіїєґ']+", text)
    return [morph.parse(t)[0].normal_form for t in tokens]

opf_top = (
    opf_df
    .assign(OPF_CODE_INT=opf_df["FIRM_OPFCD"].astype(int))
    .query("OPF_CODE_INT % 100 == 0")
    .rename(columns={"FIRM_OPFCD": "OPF_CODE", "FIRM_OPFNM": "OPF_NAME"})
)

TOP_OPF_PATTERNS = {
    100: {"підприємство", "підприємець"},
    200: {"товариство"},
    300: {"кооператив"},
    400: {"організація", "установа", "заклад"},
    500: {"об'єднання", "асоціація", "корпорація", "консорціум"},
    700: {"непідприємницький"},
    800: {"громадський", "профспілка", "партія", "релігійний", "благодійний"},
    900: {"інший"},
}


def match_top_opf(lemmas: list[str]) -> int | None:
    lemma_set = set(lemmas)

    best_code = None
    best_score = 0

    for code, keywords in TOP_OPF_PATTERNS.items():
        score = len(lemma_set & keywords)
        if score > best_score:
            best_score = score
            best_code = code

    # поріг — мінімум 1–2 збіги (регулюється)
    if best_score >= 1:
        return best_code

    return None


def extract_opf(full_name_norm, opf_ref_exact):
    # ---------- LEVEL 1: exact ----------
    for _, r in opf_ref_exact.iterrows():
        if full_name_norm.startswith(r["OPF_NAME_NORM"]):
            return r["OPF_CODE"], r["OPF_NAME"], "EXACT"

    # ---------- LEVEL 2: semantic ----------
    for code, name, markers in OPF_MARKERS:
        if any(m in full_name_norm for m in markers):
            return code, name, "SEMANTIC"

    # ---------- LEVEL 3: TOP LEVEL ----------
    lemmas = lemmatize_ua(full_name_norm)
    top_code = match_top_opf(lemmas)

    if top_code:
        name = opf_top.loc[opf_top["OPF_CODE"] == top_code, "OPF_NAME"].iloc[0]
        return top_code, name, "TOP_LEVEL"

    return None, None, "UNKNOWN"
