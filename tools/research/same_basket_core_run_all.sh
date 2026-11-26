#!/bin/bash
# same-basket-r2(80% 보유 + 20% 분할) 일괄 실행(runs/same-basket-r2/DESIGN-v1.md). r1 일괄 실행기(same_basket_run_all.sh, r4)와 같은 절차·실패 처리:
#   도구·명단·설계 사본과 해시 → 재현 기록(코드·설정·입력·DB 체크섬) → X 재현 대조(r1 10절 1, formal) → 작업(동시 PARALLEL개)
#   → 전 작업 성공 확인 → 입력·DB 재확인 → 필수 검사·조립(same_basket_core_verdict.py) → SHA256SUMS
# usage: bash tools/research/same_basket_core_run_all.sh OUT_DIR PARALLEL formal|smoke
#   OUT_DIR/jobs.tsv는 same_basket_core_jobs.py로 먼저 만든다(정식 216행: 주 108 + 대조 C 108).
#   주 작업: r1 X 실행기(same_basket_run.py)를 분할 몫 --cash 2000000으로 → same_basket_core_ledger.py --kind main(H·K·보유 몫·S).
#   대조 작업: same_basket_core_ledger.py --kind ctrl(보유 몫 + 작은 몫 → C).
#   재개 가능: 같은 명령을 다시 실행하면 성공 기록(job_status.tsv의 마지막 종료 코드 0)이 없는 작업만 다시 돈다. 재개 때는 실행 사본(code/)·명단·입력 해시가
#     처음과 같아야 한다. 실패했던 작업의 부분 산출물은 지우고 다시 만든다.
#   한 작업이 실패하면 아직 시작하지 않은 작업은 건너뛴다(.stop). 조립은 모든 작업이 성공했을 때만 한다.
#   표준 출력(작업 로그)에는 작업 번호·걸린 시간·상태만 쓴다. 하위 실행기 출력은 성공하면 지우고, 실패하면 마지막 40줄만
#     OUT_DIR/runs/<실행명>.err로 남긴다("final": 필드가 든 줄은 버림).
#   잠금(OUT_DIR/.lock)은 작업 프로세스도 물려받는다. TERM/INT를 받으면 .stop을 만들어 새 작업을 시작하지 않는다. 잠금을 잡은 뒤 실패하면
#     사유를 verdict/checks.json으로 발행하고 옛 판정·표를 거둔다(승인 목록과 대조한 원본 조립기, 아니면 셸 대체). 엔진 체크아웃이 승인 판본이고 깨끗한지 확인한다.
#   마지막 줄은 항상 SAME_BASKET_EXIT=<종료 코드>다.
#   formal: 도구·실행 사본·설계가 승인 해시 목록(APPROVED_SHA, 기본 OUT_DIR/../impl-r2.sha256)과 같아야 하고, jobs.tsv가 기대 작업표(216행)와 같아야 하며,
#     DB 체크섬을 생략할 수 없다. PRECHECK_ONLY=1이면 사전 검사만 하고 끝낸다.
#   smoke: SKIP_DB_CHECKSUM=1 허용, X 재현 대조는 RUN_REPRO=1일 때만, H·K의 r1 바이트 대조는 하지 않는다(기간이 다름).
#   DB는 SELECT/CHECKSUM만 한다. 외부 호출 없음.
set -uo pipefail
trap 'echo "SAME_BASKET_EXIT=$?"' EXIT
stop_on_signal() { [ -n "${OUT:-}" ] && [ -d "$OUT" ] && touch "$OUT/.stop"; exit "$1"; }
trap 'stop_on_signal 143' TERM; trap 'stop_on_signal 130' INT
ENGINE_HEAD=79051b59355c4e682f471f95d3166fddea29da27
R=/mnt/volume/workspace/split-research
PY=$R/env/bin/python; SRC=$R/repo/tools/research; W=$R/wt-79051b5; ETF=$R/data/etf/069500.csv
BASKETS=${BASKET_DIR:-$R/runs/same-basket-r1/baskets}; STAGE2=$R/runs/selection-vs-etf-r1/sim; R2=$R/runs/same-basket-r2; R1FULL=$R/runs/same-basket-r1/full
SPLIT_CASH=2000000
die() { echo "ERROR: $*" >&2; exit 1; }
[ $# = 3 ] || die "usage: same_basket_core_run_all.sh OUT_DIR PARALLEL formal|smoke"
OUT=$(realpath "$1"); P=$2; MODE=$3
[[ "$MODE" == formal || "$MODE" == smoke ]] || die "mode must be formal|smoke"
[[ "$P" =~ ^[1-6]$ ]] || die "PARALLEL must be 1-6"
test -f "$OUT/jobs.tsv" || die "jobs.tsv missing"
APPROVED=${APPROVED_SHA:-$(dirname "$OUT")/impl-r2.sha256}
WANT_HEAD=$ENGINE_HEAD
if [ -n "${EXPECTED_ENGINE_HEAD:-}" ]; then [ "$MODE" = smoke ] || die "formal: EXPECTED_ENGINE_HEAD override is not allowed"; WANT_HEAD=$EXPECTED_ENGINE_HEAD; fi
[ "$(git -C $W rev-parse HEAD)" = "$WANT_HEAD" ] || die "engine checkout is not the approved version"
[ -z "$(git -C $W status --porcelain)" ] || die "engine checkout has local changes"
if [ "$MODE" = formal ]; then
  [ "${SKIP_DB_CHECKSUM:-0}" = 0 ] || die "formal: DB checksum cannot be skipped"
  test -f "$APPROVED" || die "formal: approved hash list missing ($APPROVED)"
  ( cd $SRC && grep -v ' DESIGN-' "$APPROVED" | sha256sum -c --quiet - ) || die "formal: research tools differ from the approved hash list"
  ( cd $R2 && grep ' DESIGN-' "$APPROVED" | sha256sum -c --quiet - ) || die "formal: design differs from the approved hash list"
  ( cd $W && $PY -B $SRC/same_basket_core_jobs.py "$OUT" --verify ) > /dev/null || die "formal: jobs.tsv differs from the expected job table"
  test -f "$R1FULL/SHA256SUMS" || die "formal: r1 reference hash list missing"
fi
if [ "${PRECHECK_ONLY:-0}" = 1 ]; then echo "precheck ok (no run started)"; exit 0; fi
exec 9> "$OUT/.lock"; flock -n 9 || die "another run holds $OUT/.lock"
T=$OUT/code; V=$OUT/provenance; B=$OUT/baskets
PUBLISHED_NAMES="checks.json summary_jobs.csv summary_baskets.csv control.csv weight_daily.csv verdict.json VERDICT.md verdict_smoke.json smoke_tables"
retract_shell() {
  local v=$OUT/verdict reason
  for n in $PUBLISHED_NAMES; do rm -rf "${OUT:?}/$n"; done
  rm -rf "$v"; mkdir -p "$v"
  reason=$(printf '%s' "run_all: $1" | sed 's/\\/\\\\/g; s/"/\\"/g')
  printf '{\n "mode": "%s",\n "passed": false,\n "errors": [\n  "%s"\n ],\n "published_by": "run_all shell fallback"\n}\n' "$MODE" "$reason" > "$v/checks.json"
}
retract() {
  local vt=$SRC/same_basket_core_verdict.py want got
  want=$(awk '$2=="same_basket_core_verdict.py" {print $1}' "$APPROVED" 2>/dev/null); got=$(sha256sum < "$vt" | cut -d' ' -f1)
  if [ -n "$want" ] && [ "$want" = "$got" ]; then
    ( cd $W && $PY -B "$vt" "$OUT" --mode "$MODE" --fail "$1" ) > /dev/null 2>&1
    [ -f "$OUT/verdict/checks.json" ] && grep -q '"passed": false' "$OUT/verdict/checks.json" && [ ! -e "$OUT/verdict/verdict.json" ] && return 0
  fi
  retract_shell "$1"
}
die() { echo "ERROR: $*" >&2; retract "$*"; exit 1; }
die_kept() { echo "ERROR: $*" >&2; exit 1; }
TOOLS="same_basket_core_common.py same_basket_core_rules.py same_basket_core_ledger.py same_basket_core_checks.py same_basket_core_verdict.py same_basket_core_jobs.py
 same_basket_core_selftest.py same_basket_core_smoke_check.py same_basket_core_hk_repro.sh same_basket_core_run_all.sh
 same_basket_common.py same_basket_rules.py same_basket_run.py same_basket_ledger.py same_basket_aux.py same_basket_checks.py same_basket_verdict.py
 same_basket_smoke_check.py same_basket_repro.sh continuous_account_run.py split_window_runs.py stage2_common.py"
input_hashes() {
  ( cd $W && sha256sum config/config.yaml results/wfo_run_20260324_125326/final_candidate_manifest.json | sed 's#config/config.yaml#config.yaml(content hash only)#' )
  sha256sum "$(realpath $ETF)" "$OUT/jobs.tsv" $STAGE2/curves_20{14,16,18,20,22,24}.csv "$R1FULL/SHA256SUMS"
  ( cd "$B" && sha256sum basket_*.csv SHA256SUMS )
}
db_checksums() { for t in CalculatedIndicators CompanyInfo DailyStockPrice DailyStockTier FinancialData InvestorTradingTrend MarketCapDaily TickerUniverseHistory TickerUniverseSnapshot; do
  echo "$t $(mysql -uroot stocks -N -e "SELECT COUNT(*) FROM $t") $(mysql -uroot stocks -N -e "CHECKSUM TABLE $t" | awk '{print $2}')"; done; }
# 1. 실행 폴더 열기(처음) 또는 재개
if [ ! -d "$T" ]; then
  [ -z "$(ls -A "$OUT" | grep -v -x -e jobs.tsv -e .lock)" ] || die "OUT_DIR must contain only jobs.tsv on the first run"
  mkdir -p "$T" "$V" "$B" "$OUT/runs"
  for f in $TOOLS; do cp "$SRC/$f" "$T/" || die "tool missing: $f"; done
  cp "$R2/DESIGN-v1.md" "$T/" 2>/dev/null || [ "$MODE" = smoke ] || die "DESIGN-v1.md missing"
  cp "$BASKETS"/basket_*.csv "$BASKETS/SHA256SUMS" "$B/" || die "baskets missing"
  ( cd "$T" && sha256sum * > "$V/code.sha256" )
  input_hashes > "$V/inputs.sha256"
  printf 'id\tprefix\texit\tseconds\tfinished_utc\n' > "$OUT/job_status.tsv"
  echo "opened_utc $(date -u +%FT%TZ)" > "$V/env.txt"
else
  echo "resumed_utc $(date -u +%FT%TZ)" >> "$V/env.txt"
fi
( cd "$T" && sha256sum -c --quiet "$V/code.sha256" ) || die "execution copies differ from the first run (code)"
( cd "$B" && sha256sum -c --quiet SHA256SUMS ) || die "basket copies differ from their hash list"
input_hashes | diff - "$V/inputs.sha256" > "$V/inputs_check.diff" || die "inputs differ from the first run (see provenance/inputs_check.diff)"
if [ "$MODE" = formal ]; then ( cd "$T" && sha256sum -c --quiet "$APPROVED" ) || die "execution copies differ from the approved hash list"; fi
{
  echo "mode $MODE"; echo "session_utc $(date -u +%FT%TZ)"; echo "parallel $P"
  echo "repo_head $(git -C $R/repo rev-parse HEAD)"; echo "wt_head $(git -C $W rev-parse HEAD)"; echo "wt_dirty_files $(git -C $W status --porcelain | wc -l)"
  $PY -c "import sys,pandas,numpy,yaml;print('python',sys.version.split()[0],'pandas',pandas.__version__,'numpy',numpy.__version__,'pyyaml',yaml.__version__)"
  echo "universe_mode strict_pit"; echo "split_cash $SPLIT_CASH"
} >> "$V/env.txt"
# 2. DB 체크섬(기준: 수집 복구 준비 때 원본 DB 체크섬)
if [ "${SKIP_DB_CHECKSUM:-0}" != 1 ]; then
  db_checksums > "$V/db_checksum.session.txt"
  [ -f "$V/db_checksum.txt" ] || cp "$V/db_checksum.session.txt" "$V/db_checksum.txt"
  diff "$V/db_checksum.session.txt" "$V/db_checksum.txt" > "$V/db_session.diff" || die "DB differs from the first run"
  REF=$R/transfer/pre-collect-20261122/checksums_original.tsv
  awk 'NR==FNR {want[$1]=1; next} ($1 in want) {print $1, $2, $3}' "$V/db_checksum.txt" "$REF" | sort > "$V/db_reference.txt"
  diff <(sort "$V/db_checksum.txt") "$V/db_reference.txt" > "$V/db_diff.txt" || die "DB checksum differs from reference (see provenance/db_diff.txt)"
  echo "db_matches_reference yes" >> "$V/env.txt"
else
  echo "db_checksum skipped (smoke)" >> "$V/env.txt"
fi
# 3. X 재현 대조(r1 10절 1: 명단 제한을 끈 r1 X 실행기 = 연속 계좌 산출물)
if [ "$MODE" = formal ] || [ "${RUN_REPRO:-0}" = 1 ]; then
  if ! grep -q '"passed": true' "$OUT/repro/repro_check.json" 2>/dev/null; then
    rm -rf "$OUT/repro"
    BASKET_DIR="$B" bash "$T/same_basket_repro.sh" "$OUT/repro" > /dev/null 2>&1 || die "reproduction check failed (see repro/repro_check.json)"
  fi
  echo "repro ok"
fi
# 4. 작업
run_one() {
  local id=$1 kind=$2 basket=$3 k=$4 div=$5 delist=$6 ticker=$7 ratio=$8 s=$9 e=${10} pre=${11} t0 rc
  [ -e "$OUT/.stop" ] && return 0
  rm -f "$OUT/runs/${pre}"_* "$OUT/runs/$pre.log" "$OUT/runs/$pre.out" "$OUT/runs/$pre.err"
  t0=$(date +%s)
  (
    cd $W || exit 97
    export MAGICSPLIT_UNIVERSE_MODE=strict_pit
    if [ "$kind" = main ]; then
      $PY -B "$T/same_basket_run.py" "$s" "$e" "$OUT/runs/$pre" --basket "$B/basket_$basket.csv" --warmup --delist "$delist" --cash "$SPLIT_CASH" \
          --tax period --div "$div" --hold none \
        && $PY -B "$T/same_basket_core_ledger.py" "$s" "$e" "$OUT/runs/$pre" --basket "$B/basket_$basket.csv" --div "$div" --delist "$delist" --kind main --etf "$ETF"
    else
      $PY -B "$T/same_basket_core_ledger.py" "$s" "$e" "$OUT/runs/$pre" --basket "$B/basket_$basket.csv" --div "$div" --delist "$delist" --kind ctrl --etf "$ETF"
    fi
  ) > "$OUT/runs/$pre.out" 2>&1 && rc=0 || rc=$?
  [ "$rc" = 0 ] || tail -n 40 "$OUT/runs/$pre.out" | grep -v '"final":' > "$OUT/runs/$pre.err"
  rm -f "$OUT/runs/$pre.out"
  printf '%s\t%s\t%s\t%s\t%s\n' "$id" "$pre" "$rc" "$(( $(date +%s) - t0 ))" "$(date -u +%FT%TZ)" >> "$OUT/job_status.tsv"
  if [ "$rc" = 0 ]; then echo "job $id $(( $(date +%s) - t0 ))s ok"; else touch "$OUT/.stop"; echo "job $id $(( $(date +%s) - t0 ))s FAILED($rc)"; fi
}
export -f run_one; export OUT PY T W B ETF SPLIT_CASH
rm -f "$OUT/.stop"
pending() { awk -F'\t' 'NR==FNR { if (FNR>1) last[$2]=$3; next } FNR>1 && last[$11] != "0"' "$OUT/job_status.tsv" "$OUT/jobs.tsv"; }
echo "jobs_total $(( $(wc -l < "$OUT/jobs.tsv") - 1 )) pending $(pending | wc -l)"
pending | xargs -r -P "$P" -L1 bash -c 'run_one "$@"' _ &
XP=$!; wait $XP || true
echo "jobs_finished_utc $(date -u +%FT%TZ)" >> "$V/env.txt"
[ ! -e "$OUT/.stop" ] || die "a job failed; remaining jobs were skipped (see job_status.tsv); no verdict"
[ "$(pending | wc -l)" = 0 ] || die "some jobs are not finished (see job_status.tsv); no verdict"
# 5. 실행 중 입력 변경 확인
input_hashes | diff - "$V/inputs.sha256" > "$V/inputs_recheck.diff" || die "inputs changed during the run (see provenance/inputs_recheck.diff)"
if [ "${SKIP_DB_CHECKSUM:-0}" != 1 ]; then
  db_checksums | diff - "$V/db_checksum.txt" > "$V/db_recheck.diff" || die "DB changed during the run (see provenance/db_recheck.diff)"
  echo "db_recheck_after_jobs yes" >> "$V/env.txt"
fi
# 6. 필수 검사·조립(실패 시 판정 미발행)
( cd $W && $PY -B "$T/same_basket_core_verdict.py" "$OUT" --mode "$MODE" --etf "$ETF" ) > "$OUT/verdict.log" 2>&1 || die_kept "checks failed; no verdict (see verdict/checks.json)"
echo "finished_utc $(date -u +%FT%TZ)" >> "$V/env.txt"
( cd "$OUT" && find . -type f ! -name SHA256SUMS ! -name '.lock' | sort | xargs sha256sum > SHA256SUMS )
echo "done: $OUT"
