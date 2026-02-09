dfs = []

for arcdate in arcdates:
    temp = get_data(
        QUERY,
        sql_params={"arcdate": arcdate}
    )

    # фільтримо тільки потрібні компанії
    temp = temp[temp["CONTRAGENTID_ZP"].isin(base_clients["IDENTIFYCODE"])]

    print(f"Doing {arcdate}, shape: {temp.shape[0]}")
    dfs.append(temp)

zkp_employees = pd.concat(dfs, ignore_index=True)
