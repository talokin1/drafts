select
    max(balance_amt_uah) as max_balance_amt_uah
from b2_olap.ar_deals@dwh
where contragentid = 3050596
  and arcdate = date '2025-11-30';
