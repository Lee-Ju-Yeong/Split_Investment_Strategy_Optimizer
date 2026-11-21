
"""PIT 표본의 KRX 상태 전환일을 DART 사업보고서 최초·정정 접수일, 원문 주요계정과 대조한다(읽기 전용 조회)."""
import io, json, time, zipfile, re, sys
import xml.etree.ElementTree as ET
import pandas as pd, requests, yaml
# usage: python addbuy_pit_dart_check.py SAMPLE.csv OUT.json [CONFIG.yaml]  (키는 config의 dart.api_key, 출력하지 않음)
SAMPLE, OUT = sys.argv[1], sys.argv[2]
K = yaml.safe_load(open(sys.argv[3] if len(sys.argv) > 3 else 'config/config.yaml', encoding='utf-8'))['dart']['api_key']
B = 'https://opendart.fss.or.kr/api/'
def get(ep, **p):
    time.sleep(0.15); r = requests.get(B + ep, params={'crtfc_key': K, **p}, timeout=30); return r
z = zipfile.ZipFile(io.BytesIO(get('corpCode.xml').content))
root = ET.fromstring(z.read(z.namelist()[0]))
cmap = {e.findtext('stock_code').strip(): e.findtext('corp_code') for e in root.iter('list') if (e.findtext('stock_code') or '').strip()}
s = pd.read_csv(SAMPLE, dtype={'ticker': str})
def lst(cc, bgn, end, ty=None):
    out, page = [], 1
    while True:
        p = dict(corp_code=cc, bgn_de=bgn, end_de=end, page_no=page, page_count=100)
        if ty: p['pblntf_ty'] = ty
        j = get('list.json', **p).json()
        if j.get('status') != '000': return out
        out += j['list']
        if page >= int(j.get('total_page', 1)): return out
        page += 1
def num(x):
    try: return float(str(x).replace(',', ''))
    except: return None
def accounts(cc, fy):
    j = get('fnlttSinglAcntAll.json', corp_code=cc, bsns_year=str(fy), reprt_code='11011', fs_div='CFS').json()
    fs = 'CFS'
    if j.get('status') != '000':
        j = get('fnlttSinglAcntAll.json', corp_code=cc, bsns_year=str(fy), reprt_code='11011', fs_div='OFS').json(); fs = 'OFS'
    if j.get('status') != '000': return {'fs': None, 'msg': j.get('message')}
    rows = j['list']
    def pick(pred):
        for r in rows:
            if pred(r): return num(r.get('thstrm_amount')), num(r.get('frmtrm_amount'))
        return None, None
    nm = lambda r: re.sub(r'\s', '', r.get('account_nm', ''))
    IS = lambda r: r.get('sj_div') in ('IS', 'CIS')
    ni = pick(lambda r: IS(r) and r.get('account_id') == 'ifrs-full_ProfitLossAttributableToOwnersOfParent')
    if ni[0] is None: ni = pick(lambda r: IS(r) and '지배' in nm(r) and ('순이익' in nm(r) or '순손익' in nm(r)) and '비지배' not in nm(r))
    if ni[0] is None: ni = pick(lambda r: IS(r) and r.get('account_id') == 'ifrs-full_ProfitLoss')
    if ni[0] is None: ni = pick(lambda r: IS(r) and nm(r) in ('당기순이익', '당기순이익(손실)', '당기순손익'))
    BS = lambda r: r.get('sj_div') == 'BS'
    eq = pick(lambda r: BS(r) and r.get('account_id') == 'ifrs-full_EquityAttributableToOwnersOfParent')
    if eq[0] is None: eq = pick(lambda r: BS(r) and '지배기업' in nm(r) and '지분' in nm(r) and '비지배' not in nm(r))
    if eq[0] is None: eq = pick(lambda r: BS(r) and (r.get('account_id') == 'ifrs-full_Equity' or nm(r) == '자본총계'))
    eps = pick(lambda r: r.get('account_id') == 'ifrs-full_BasicEarningsLossPerShare' or ('기본' in nm(r) and '주당' in nm(r)))
    return {'fs': fs, 'ni': ni, 'eq': eq, 'eps': eps}
res = []
for _, r in s.iterrows():
    cc = cmap.get(r.ticker); fd = pd.Timestamp(r.first_date); sd = pd.Timestamp(r.signal_date)
    row = {'ticker': r.ticker, 'state': r.state_key, 'krx_switch': str(fd.date()), 'signal_date': str(sd.date()),
           'krx_eps0_eps1': [r.eps0, r.eps], 'krx_roe0_roe1': [round(r.roe0,2), round(r.roe1,2)], 'corp_code': cc}
    if cc is None: row['note'] = 'corp_code 없음'; res.append(row); continue
    switch_like = fd.month in (4, 5)
    if switch_like:
        fy = fd.year - 1
        L = lst(cc, f'{fy+1}0101', f'{fy+3}1231', 'A')
        ann = [x for x in L if '사업보고서' in x['report_nm'] and f'({fy}.' in x['report_nm']]
        orig = sorted([x for x in ann if '정정' not in x['report_nm'] and '첨부' not in x['report_nm']], key=lambda x: x['rcept_dt'])
        amend = sorted([x for x in ann if '정정' in x['report_nm']], key=lambda x: x['rcept_dt'])
        row.update({'fy': fy, 'first_filing': orig[0]['rcept_dt'] if orig else None,
                    'amendments': [(x['rcept_dt'], x['report_nm']) for x in amend]})
        if orig:
            ff = pd.Timestamp(orig[0]['rcept_dt']); row['filed_before_switch'] = bool(ff <= fd); row['days_filing_to_switch'] = (fd - ff).days
            row['amend_before_signal'] = [a[0] for a in row['amendments'] if pd.Timestamp(a[0]) <= sd]
            row['amend_after_signal'] = [a[0] for a in row['amendments'] if pd.Timestamp(a[0]) > sd]
        a = accounts(cc, fy) if fy >= 2015 else {'fs': None, 'msg': 'DART 재무 API는 2015 사업연도부터'}
        row['dart'] = a
        if a.get('fs') and None not in (a['ni'][0], a['ni'][1], a['eq'][0], a['eq'][1]) and a['eq'][0] > 0 and a['eq'][1] > 0:
            roe_t, roe_p = 100*a['ni'][0]/a['eq'][0], 100*a['ni'][1]/a['eq'][1]
            row['dart_roe_prev_cur'] = [round(roe_p,2), round(roe_t,2)]
            row['direction_match'] = bool((roe_t < roe_p) == (r.roe1 < r.roe0))
    else:
        L = lst(cc, (fd - pd.Timedelta(days=120)).strftime('%Y%m%d'), (fd + pd.Timedelta(days=5)).strftime('%Y%m%d'))
        kw = re.compile('무상|증자|분할|병합|감자|주식배당|합병|소각')
        row['capital_events'] = [(x['rcept_dt'], x['report_nm'].strip()) for x in L if kw.search(x['report_nm'])][:8]
        row['eps_bps_ratio'] = [round(r.eps/r.eps0, 4), round(r.bps/r.bps0, 4)]
        A = lst(cc, (fd - pd.Timedelta(days=120)).strftime('%Y%m%d'), (fd + pd.Timedelta(days=5)).strftime('%Y%m%d'), 'A')
        row['periodic_reports'] = [(x['rcept_dt'], x['report_nm'].strip()) for x in A][:6]
    res.append(row)
json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=str)
for x in res: print(json.dumps(x, ensure_ascii=False, default=str))
