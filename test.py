FEATURE_MAP = {}
FEATURE_MAP.update(BUSINESS_MAP)
FEATURE_MAP.update(PORTFOLIO_MAP)
FEATURE_MAP.update(DAILY_MAP)

import re

VALID_FEATURE_RE = re.compile(r'^(BUSINESS_B\d+|PORTFOLIO_P\d+|DAILY_D\d+)$')

cluster_feature_impact = cluster_feature_impact[
    cluster_feature_impact['FEATURE'].str.match(VALID_FEATURE_RE)
].copy()

def decode_feature(feature, mapping):
    return mapping.get(feature, feature)

cluster_feature_impact['FEATURE_TXT'] = cluster_feature_impact['FEATURE'].apply(
    lambda x: decode_feature(x, FEATURE_MAP)
)
cluster_feature_impact = (
    cluster_feature_impact
    .sort_values(['Кластер','cnt'], ascending=[True, False])
    [['Кластер','BLOCK','FEATURE','FEATURE_TXT','cnt']]
)
