"""k=0 재생성 패널을 panel_v2와 거래일·신호일·코드·순위로 대조한다(v2.1 B: 정식 전 144개월 전수).

usage: python -I -B tools/research/pool_vs_index_panel_check.py NEW_PANEL.csv PANEL_v2.csv OUT.json [--full]
--full: 144개월·전체 행(9,875) 모두 같아야 통과. 없으면 새 패널에 있는 달만 대조한다(smoke).
"""
import json, os, sys
import pandas as pd

KEYS = ['trade_date', 'signal_date', 'ticker', 'rank']
FULL_MONTHS, FULL_ROWS = 144, 9875


def load(p):
    return pd.read_csv(p, dtype={'ticker': str})[KEYS].astype(str).sort_values(KEYS).reset_index(drop=True)


def main():
    new_p, ref_p, out = sys.argv[1:4]; full = '--full' in sys.argv[4:]
    new, ref = load(new_p), load(ref_p)
    if not full: ref = ref[ref.trade_date.isin(set(new.trade_date))].reset_index(drop=True)
    m = new.merge(ref, how='outer', indicator=True)
    res = {'mode': 'full' if full else 'months_in_new', 'new_rows': len(new), 'ref_rows': len(ref), 'new_months': new.trade_date.nunique(),
           'ref_months': ref.trade_date.nunique(), 'only_new': int((m._merge == 'left_only').sum()), 'only_ref': int((m._merge == 'right_only').sum())}
    res['pass'] = res['only_new'] == 0 and res['only_ref'] == 0 and len(new) == len(ref) and len(new) > 0
    if full: res['pass'] = res['pass'] and res['new_months'] == FULL_MONTHS and res['new_rows'] == FULL_ROWS
    res['first_diffs'] = m[m._merge != 'both'].head(20).astype(str).to_dict('records')
    if os.path.exists(out): raise SystemExit(f'refuse to overwrite {out}')
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'first_diffs'}, indent=1))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
