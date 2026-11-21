
"""C1 체결 상태에서 공개 시점 점검 표본을 고정 시드로 뽑는다: 연도별 일괄 갱신 1건 + 그 밖의 날짜 4건.
usage: python addbuy_pit_sample.py RESEARCH_LABELED.csv HOLDOUT_LABELED.csv OUT.csv"""
import sys
import pandas as pd, numpy as np
fr = []
for f, period in [(sys.argv[1],'2014-2023'), (sys.argv[2],'2024-2025')]:
    d = pd.read_csv(f, dtype={'ticker':str}); d['period']=period; fr.append(d)
d = pd.concat(fr); d = d[d.C1 & d.filled]
s = d.sort_values('signal_date').groupby('state_key').first().reset_index()
s['s1_month'] = pd.to_datetime(s.first_date).dt.month; s['s1_year'] = pd.to_datetime(s.first_date).dt.year
rng = np.random.default_rng(20261121)
fd = pd.to_datetime(s.first_date)
allst = pd.concat(fr); allst = allst.drop_duplicates('state_key'); allst['fd'] = pd.to_datetime(allst.first_date)
switch_day = allst.groupby(allst.fd.dt.year).fd.agg(lambda x: x.value_counts().idxmax())
print('annual switch day by year:', {int(k): str(v.date()) for k, v in switch_day.items()})
s['is_switch'] = fd.values == s.s1_year.map(switch_day).values
may = s[s.is_switch]; off = s[~s.is_switch]
pick_may = may.groupby('s1_year', group_keys=False).apply(lambda g: g.sample(1, random_state=int(rng.integers(1e9))))
pick_off = off.sample(min(4, len(off)), random_state=20261121)
p = pd.concat([pick_may, pick_off])
cols = ['ticker','state_key','s0_first','eps0','bps0','first_date','eps','bps','roe0','roe1','signal_date','order_date','period']
print('C1 filled states:', len(s), 'may:', len(may), 'off-cycle:', len(off))
print(p[cols].to_string(index=False))
p[cols].to_csv(sys.argv[3], index=False)
