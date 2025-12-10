banks_final = bank_clients.merge(
    bank_txn[["bank_name", "n_txn", "total_sum"]],
    on="bank_name",
    how="left"
)

# 🔥 КЛЮЧОВЕ — прибираємо дублікати, які породив merge
banks_final = (
    banks_final
    .groupby("bank_name", as_index=False)
    .agg(
        clients=("clients", "max"),        # бо значення однакові
        n_txn=("n_txn", "sum"),            # сумуємо транзакції
        total_sum=("total_sum", "sum")     # сумуємо суми
    )
)

banks_final = banks_final.sort_values("clients", ascending=False)
banks_final
