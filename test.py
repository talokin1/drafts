# 1. ЄДРПОУ з clients
clients_set = set(clients["IDENTIFYCODE"].astype(str))

# 2. ЄДРПОУ, у яких реально є еквайринг у df
acq_set = set(
    df.loc[df["Ознака екв"] == "екв", "CONTRAGENTBIDENTIFYCODE"]
      .astype(str)
)

# 3. Клієнти без еквайрингу, але які потрапили в clients
clients_without_acq = clients_set - acq_set

len(clients_without_acq)
