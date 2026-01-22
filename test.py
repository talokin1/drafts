ALL_MAP_MARKERS = [
    MAP_MARKERS_100,
    MAP_MARKERS_200,
    MAP_MARKERS_300,
    MAP_MARKERS_400,
    MAP_MARKERS_500,
    MAP_MARKERS_700,
    MAP_MARKERS_800,
    MAP_MARKERS_900,
]


def extract_opf(full_name_norm: str):
    # ---------- LEVEL 1: exact ----------
    for _, r in opf_ref_exact.iterrows():
        if full_name_norm.startswith(r["OPF_NAME_NORM"]):
            return r["OPF_CODE"], r["OPF_NAME"], "EXACT"

    # ---------- LEVEL 2: marker-based ----------
    for marker_group in ALL_MAP_MARKERS:
        for opf_code, patterns in marker_group.items():
            for pattern in patterns:
                if re.search(pattern, full_name_norm):
                    name = opf_ref.loc[
                        opf_ref["OPF_CODE_INT"] == opf_code, "OPF_NAME"
                    ].iloc[0]
                    return opf_code, name, "MARKER"

    return None, None, "UNKNOWN"
