#!/bin/bash
# 연구 (b) 1등급 후보군 대 시장·스타일 지수 일괄 실행(DESIGN-v2.md + DESIGN-v2.1.md + 구현 리뷰 F1~F10).
# usage: bash tools/research/pool_vs_index_run_all.sh OUT_DIR smoke|formal
#   smoke: START=2023-12-01 END=2024-01-31 CUT=2023-12-28 기본. FULL_PANEL_CHECK=1(기본)이면 k=0 패널 144개월 전수 재생성·대조도 한다.
#          SKIP_DB_CHECKSUM=1이면 DB 체크섬을 건너뛴다(smoke만).
#   formal: 2014-01-01~2025-12-30, CUT=2023-12-28. OUT_DIR/../approved/{code.sha256,design.sha256,inputs.sha256}가 있어야 하고
#           다르면 멈춘다(F4: 보관 패널·ETF·2단계 기준 파일 포함). code.sha256은 연구 소스 14개, design.sha256은
#           DESIGN-v2.md·DESIGN-v2.1.md·DESIGN-v2.2-note.md를 정확히 담아야 한다. k=0 144개월 전수 대조가 실패하면 즉시 멈춘다.
# 순서: 사본·해시 → 자체 검사(--db) → DB 체크섬(전) → 실행별 ETF 입력(END·CUT) → 패널(k0,k5) → 패널 대조
#       → 절단 패널(k0,k5; 패널부터 다시) → 자료 묶음 4개 → 모의 실행 4개 → 절단 검사(k0,k5) → S20 재현 → DB 체크섬(후) → 판정 → SHA256SUMS
# 동시 실행: 패널 생성만 2개, 자료 묶음·모의 실행은 순차(메모리; formal은 systemd-run MemoryMax=24G 권장). DB는 SELECT·CHECKSUM만, 외부 시세 호출 없음. 폴더는 새 폴더만(.opened, noclobber).
set -euo pipefail
R=/mnt/volume/workspace/split-research
PY=$R/env/bin/python; SRC=$R/repo/tools/research; W=$R/wt-79051b5; ETFD=$R/data/etf; STAGE2=$R/runs/selection-vs-etf-r1
DBREF=$R/transfer/pre-collect-20261122/checksums_original.tsv
DB_TABLES="CalculatedIndicators CompanyInfo DailyStockPrice DailyStockTier FinancialData InvestorTradingTrend MarketCapDaily TickerUniverseHistory TickerUniverseSnapshot"
OUT=$(realpath -m "$1"); MODE=$2
die() { echo "ERROR: $*" >&2; exit 1; }
[[ "$MODE" == smoke || "$MODE" == formal ]] || die "mode must be smoke|formal"
if [ "$MODE" = formal ]; then START=2014-01-01; END=2025-12-30; CUT=2023-12-28; FULL_PANEL_CHECK=0; SKIP_DB_CHECKSUM=0
else START=${START:-2023-12-01}; END=${END:-2024-01-31}; CUT=${CUT:-2023-12-28}; FULL_PANEL_CHECK=${FULL_PANEL_CHECK:-1}; SKIP_DB_CHECKSUM=${SKIP_DB_CHECKSUM:-0}; fi
input_hashes() {
  ( cd $W && sha256sum config/config.yaml results/wfo_run_20260324_125326/final_candidate_manifest.json | sed 's#config/config.yaml#config.yaml(content hash only)#' )
  sha256sum $ETFD/069500.csv $ETFD/069500_quality.csv $ETFD/229200.csv $ETFD/226980.csv $ETFD/252650.csv $ETFD/277650.csv \
    $STAGE2/panel_v2.csv $STAGE2/sim_v2/curves_2014.csv $STAGE2/sim_v2/accounts_2014.csv
}
AP=$(dirname "$OUT")/approved
SOURCES="pool_vs_index_accounting.py pool_vs_index_common.py pool_vs_index_data.py pool_vs_index_etf_cap.py pool_vs_index_etf_extract.py pool_vs_index_legacy_s20.py pool_vs_index_panel.py pool_vs_index_panel_check.py pool_vs_index_run_all.sh pool_vs_index_selftest.py pool_vs_index_sim.py pool_vs_index_truncation_check.py pool_vs_index_verdict.py stage2_common.py"
DESIGNS="DESIGN-v2.md DESIGN-v2.1.md DESIGN-v2.2-note.md"
listed() { awk '{sub(/^\*/, "", $2); print $2}' "$1" | sort | tr '\n' ' '; }
want() { echo $1 | tr ' ' '\n' | sort | tr '\n' ' '; }
if [ "$MODE" = formal ]; then
  [ "$(listed "$AP/code.sha256")" = "$(want "$SOURCES")" ] || die "approved code.sha256 must list exactly the 14 research sources"
  [ "$(listed "$AP/design.sha256")" = "$(want "$DESIGNS")" ] || die "approved design.sha256 must list DESIGN-v2.md, DESIGN-v2.1.md, DESIGN-v2.2-note.md"
  ( cd $SRC && sha256sum -c --strict --quiet "$AP/code.sha256" ) || die "research tools differ from approved (or malformed line)"      # H4
  ( cd "$(dirname "$OUT")" && sha256sum -c --strict --quiet "$AP/design.sha256" ) || die "design differs from approved (or malformed line)"
  input_hashes | diff - "$AP/inputs.sha256" > /dev/null || die "inputs (panel_v2, ETF, stage2 refs, config hash) differ from approved"
fi
[ ! -e "$OUT" ] || [ -z "$(ls -A "$OUT")" ] || die "OUT_DIR must be new or empty"
mkdir -p "$OUT"; ( set -o noclobber; date -u +%FT%TZ > "$OUT/.opened" ) 2>/dev/null || die "OUT_DIR already opened"
mkdir -p "$OUT/code" "$OUT/provenance" "$OUT/panel" "$OUT/checks" "$OUT/logs"
for f in $SOURCES; do cp "$SRC/$f" "$OUT/code/"; done
mkdir -p "$OUT/design"
for d in $DESIGNS; do f=$(dirname "$OUT")/$d; [ -f "$f" ] || f=$(dirname "$(dirname "$OUT")")/$d; [ -f "$f" ] || die "design file $d missing"; cp "$f" "$OUT/design/"; done
( cd "$OUT/code" && sha256sum $SOURCES > "$OUT/provenance/code.sha256" )       # 승인 목록과 같은 모양: 연구 소스 14개
( cd "$OUT/design" && sha256sum $DESIGNS > "$OUT/provenance/design.sha256" )   # 설계 문서 3개(v2, v2.1, v2.2-note)
T=$OUT/code; V=$OUT/provenance; C=$OUT/checks; L=$OUT/logs
{
  echo "mode $MODE"; echo "start $START"; echo "end $END"; echo "cut $CUT"; echo "started_utc $(date -u +%FT%TZ)"
  echo "repo_head $(git -C $R/repo rev-parse HEAD)"; echo "repo_dirty_files $(git -C $R/repo status --porcelain | wc -l)"
  echo "wt_head $(git -C $W rev-parse HEAD)"; echo "wt_dirty_files $(git -C $W status --porcelain | wc -l)"
  $PY -c "import sys,pandas,numpy;print('python',sys.version.split()[0],'pandas',pandas.__version__,'numpy',numpy.__version__)"
} > "$V/env.txt"
input_hashes > "$V/inputs.sha256"
py() { ( cd $W && MAGICSPLIT_UNIVERSE_MODE=strict_pit $PY -B "$@" ); }
ensure_json() {   # R3: 하위 검사가 JSON 없이 끝나면 실패 사유 JSON을 만든다
  [ -s "$1" ] || printf '{"pass": false, "reason": "%s exited without writing its result"}\n' "$2" > "$1"; }
db_checksums() { for t in $DB_TABLES; do echo "$t $(mysql -uroot stocks -N -e "SELECT COUNT(*) FROM $t") $(mysql -uroot stocks -N -e "CHECKSUM TABLE $t" | awk '{print $2}')"; done; }
db_reference() { awk -v want="$DB_TABLES" 'BEGIN {n=split(want,a," "); for(i=1;i<=n;i++) w[a[i]]=1} ($1 in w) {print $1, $2, $3}' "$DBREF" | sort; }
# 1. 자체 검사
py $T/pool_vs_index_selftest.py --db > "$C/selftest.log" 2>&1 || die "selftest failed (checks/selftest.log)"
# 2. DB 체크섬(전)
if [ "$SKIP_DB_CHECKSUM" != 1 ]; then
  db_checksums | sort > "$V/db_checksum_before.txt"
  diff "$V/db_checksum_before.txt" <(db_reference) > "$V/db_diff_before.txt" || die "DB checksum differs from reference"
fi
# 3. 실행별 ETF 입력(날짜 필드만 보고 상한 이하 줄만 복사)
$PY -I -B $T/pool_vs_index_etf_cap.py $ETFD "$OUT/etf_full" 2025-12-30 > "$L/etf_full.log" 2>&1 || die "etf_full"
$PY -I -B $T/pool_vs_index_etf_cap.py $ETFD "$OUT/etf_cut" $CUT > "$L/etf_cut.log" 2>&1 || die "etf_cut"
# 4. 패널(동시 2개)
panel() { py $T/pool_vs_index_panel.py "$1" "$2" "$OUT/panel/$3.csv" --offset "$4" > "$L/panel_$3.log" 2>&1; }
panel $START $END panel_k0 0 & P0=$!; panel $START $END panel_k5 5 & P5=$!
wait $P0 || die "panel k0 failed"; wait $P5 || die "panel k5 failed"
# 5. 패널 대조. formal은 기간 전체(=144개월) 전수이고 실패하면 즉시 중단(F4)
if [ "$MODE" = formal ]; then
  $PY -I -B $T/pool_vs_index_panel_check.py "$OUT/panel/panel_k0.csv" $STAGE2/panel_v2.csv "$C/panel_check.json" --full > "$L/panel_check.log" 2>&1 \
    || { ensure_json "$C/panel_check.json" panel_check; die "k=0 144-month panel check failed; stopping before simulation"; }
else
  $PY -I -B $T/pool_vs_index_panel_check.py "$OUT/panel/panel_k0.csv" $STAGE2/panel_v2.csv "$C/panel_check_smoke.json" > "$L/panel_check_smoke.log" 2>&1 \
    || echo "smoke panel check failed" >> "$V/env.txt"
  if [ "$FULL_PANEL_CHECK" = 1 ]; then
    panel 2014-01-01 2025-12-30 panel_k0_144m 0 || die "panel 144m failed"
    $PY -I -B $T/pool_vs_index_panel_check.py "$OUT/panel/panel_k0_144m.csv" $STAGE2/panel_v2.csv "$C/panel_check.json" --full > "$L/panel_check.log" 2>&1 \
      || echo "panel 144m check failed" >> "$V/env.txt"
    ensure_json "$C/panel_check.json" panel_check
  else cp "$C/panel_check_smoke.json" "$C/panel_check.json"; fi
fi
# 6. 절단 패널(k0,k5: 상한 CUT로 패널부터 다시)
panel $START $CUT panel_k0_cut 0 & P0=$!; panel $START $CUT panel_k5_cut 5 & P5=$!
wait $P0 || die "panel k0_cut failed"; wait $P5 || die "panel k5_cut failed"
# 7. 자료 묶음, 8. 모의 실행
bundle() { py $T/pool_vs_index_data.py "$OUT/panel/panel_$1.csv" "$OUT/bundle_$1" --offset "$2" --start $START --end "$3" --cap "$4" > "$L/bundle_$1.log" 2>&1; }
# 메모리: 자료 묶음·모의 실행은 하나씩 순차(패널 생성만 2개 동시)
bundle k0 0 $END 2025-12-30 || die "bundle k0"; bundle k5 5 $END 2025-12-30 || die "bundle k5"
bundle k0_cut 0 $CUT $CUT || die "bundle k0_cut"; bundle k5_cut 5 $CUT $CUT || die "bundle k5_cut"
sim() { py $T/pool_vs_index_sim.py "$OUT/bundle_$1" "$OUT/sim_$1" --etf-dir "$2" > "$L/sim_$1.log" 2>&1; }
for s in k0 k5; do sim $s "$OUT/etf_full" || die "sim $s"; done
for s in k0_cut k5_cut; do sim $s "$OUT/etf_cut" || die "sim $s"; done
# 9. 절단 검사(k0,k5), 10. S20 재현
for k in k0 k5; do
  $PY -I -B $T/pool_vs_index_truncation_check.py "$OUT/sim_$k" "$OUT/sim_${k}_cut" "$C/truncation_$k.json" --cut $CUT --full-end $END --full-cap 2025-12-30 --k ${k#k} --start $START \
    --panels "$OUT/panel/panel_$k.csv" "$OUT/panel/panel_${k}_cut.csv" > "$L/truncation_$k.log" 2>&1 || echo "truncation $k failed" >> "$V/env.txt"
  ensure_json "$C/truncation_$k.json" "truncation_$k"
done
py $T/pool_vs_index_legacy_s20.py $STAGE2/panel_v2.csv $STAGE2 "$C/stage2_s20.json" > "$L/stage2_s20.log" 2>&1 || echo "stage2 s20 failed" >> "$V/env.txt"
ensure_json "$C/stage2_s20.json" stage2_s20
# 11. 입력·DB 재확인
input_hashes | diff - "$V/inputs.sha256" > "$V/inputs_recheck.diff" || die "inputs changed during the run"
if [ "$SKIP_DB_CHECKSUM" != 1 ]; then
  db_checksums | sort > "$V/db_checksum_after.txt"
  if diff "$V/db_checksum_before.txt" "$V/db_checksum_after.txt" > "$V/db_recheck.diff"; then echo '{"pass": true}' > "$C/db.json"; else echo '{"pass": false}' > "$C/db.json"; fi
else echo '{"pass": false, "skipped": true}' > "$C/db.json"; fi
# 12. 판정(검산 실패는 판정기가 undetermined로 남긴다)
$PY -I -B $T/pool_vs_index_verdict.py "$OUT/verdict" --sim k0="$OUT/sim_k0" --sim k5="$OUT/sim_k5" --mode $MODE \
  --check panel="$C/panel_check.json" --check truncation_k0="$C/truncation_k0.json" --check truncation_k5="$C/truncation_k5.json" \
  --check stage2_s20="$C/stage2_s20.json" --check db="$C/db.json" > "$L/verdict.log" 2>&1 || die "verdict failed"
echo "finished_utc $(date -u +%FT%TZ)" >> "$V/env.txt"
( cd "$OUT" && find . -type f ! -name SHA256SUMS ! -name '.opened' | sort | xargs sha256sum > SHA256SUMS )
echo "done: $OUT"
