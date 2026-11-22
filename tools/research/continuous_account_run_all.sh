#!/bin/bash
# 연속 계좌 진단(a) 일괄 실행(DESIGN-v2.2.md 7절). 한 서비스 안에서 순서대로:
#   설계·도구 사본과 해시 → 재현 기록(코드·설정·입력·DB 체크섬) → 3월 재현(anchor) 먼저 실행·확인 → 나머지 작업(동시 PARALLEL개)
#   → 전 작업 성공 확인 → 2단계 대조(판정 전) → 사본 도구로 필수 검사·집계·판정 → 최종 산출물 SHA256SUMS
#   formal은 시작 전에 승인 해시(OUT_DIR/../approved: 작업표·ETF·2단계 원본 46개·설계·도구)를 대조하고 다르면 멈춘다.
#   실행 폴더를 연 뒤 approved/를 provenance/approved/로 복사하고, 실행 사본(code/)·작업표·ETF·2단계 원본을 그 복사본 목록으로 다시 검증한다.
#   이후 대조·집계에는 복사본 목록 경로만 넘긴다(원본 approved/를 다시 읽지 않음). 작업 뒤 설정·manifest·ETF 해시와 DB 체크섬을 다시 확인한다.
#   시험용(smoke 전용): TEST_HOOK_AFTER_COPY='명령'(사본 생성 직후 실행), STOP_AFTER_COPY_VERIFY=1(사본 검증 뒤 종료).
# usage: bash tools/research/continuous_account_run_all.sh OUT_DIR PARALLEL formal|smoke
#   OUT_DIR/jobs.tsv는 continuous_account_jobs.py로 먼저 만든다. 폴더는 한 번만 연다(.opened 표시).
#   formal: PARALLEL 1~3, DB 체크섬 생략 불가, OUT_DIR/../DESIGN-v2.2.md와 approved/ 필요. smoke: SKIP_DB_CHECKSUM=1 허용, anchor 생략 가능.
#   폴더에는 jobs.tsv만 있어야 하며, 사전 검사 통과 뒤 .opened를 원자적으로(noclobber) 만든다.
#   PRECHECK_ONLY=1 이면 사전 검사(승인 해시 등)만 하고 실행 없이 끝낸다.
#   DB는 SELECT/CHECKSUM만 한다. 외부 시세 호출 없음.
set -euo pipefail
R=/mnt/volume/workspace/split-research
PY=$R/env/bin/python; SRC=$R/repo/tools/research; W=$R/wt-79051b5; ETF=$R/data/etf/069500.csv; STAGE2=$R/runs/selection-vs-etf-r1
OUT=$(realpath "$1"); P=$2; MODE=$3
die() { echo "ERROR: $*" >&2; exit 1; }
[[ "$MODE" == formal || "$MODE" == smoke ]] || die "mode must be formal|smoke"
[[ "$P" =~ ^[1-9]$ ]] || die "PARALLEL must be 1-9"
test -f "$OUT/jobs.tsv" || die "jobs.tsv missing"
DESIGN=$(dirname "$OUT")/DESIGN-v2.2.md; AP=$(dirname "$OUT")/approved
[ "$(ls -A "$OUT")" = "jobs.tsv" ] || die "OUT_DIR must contain only jobs.tsv (fresh folder)"
if [ "$MODE" = formal ]; then
  (( P <= 3 )) || die "formal: PARALLEL must be <= 3"
  [ "${SKIP_DB_CHECKSUM:-0}" = 0 ] || die "formal: DB checksum cannot be skipped"
  test -f "$DESIGN" || die "formal: DESIGN-v2.2.md missing"
  [ -z "${TEST_HOOK_AFTER_COPY:-}" ] && [ "${STOP_AFTER_COPY_VERIFY:-0}" = 0 ] || die "formal: test hooks are not allowed"
  [ "$(sha256sum < "$OUT/jobs.tsv" | cut -d' ' -f1)" = "$(awk '$2=="jobs.tsv"{print $1}' "$AP/SHA256SUMS")" ] || die "jobs.tsv differs from approved"
  ( cd "$AP" && grep -v ' jobs.tsv$' SHA256SUMS | sha256sum -c --quiet - ) || die "ETF differs from approved"
  sha256sum -c --quiet "$AP/stage2_sources.sha256" || die "stage2 sources differ from approved"
  ( cd $SRC && sha256sum -c --quiet "$AP/code.sha256" ) || die "research tools differ from approved"
  ( cd "$(dirname "$OUT")" && sha256sum -c --quiet "$AP/design.sha256" ) || die "DESIGN-v2.2.md differs from approved"
fi
if [ "${PRECHECK_ONLY:-0}" = 1 ]; then echo "precheck ok (no run started)"; exit 0; fi
( set -o noclobber; date -u +%FT%TZ > "$OUT/.opened" ) 2>/dev/null || die "OUT_DIR already opened by another run"
mkdir -p "$OUT/runs" "$OUT/provenance" "$OUT/code"
# 1. 설계·도구 사본(이후 실행·집계는 사본으로)
cp $SRC/continuous_account_*.py $SRC/continuous_account_run_all.sh $SRC/split_window_runs.py $SRC/stage2_common.py "$OUT/code/"
[ -f "$DESIGN" ] && cp "$DESIGN" "$OUT/code/"
[ -d "$AP" ] && cp -r "$AP" "$OUT/provenance/approved"
[ -n "${TEST_HOOK_AFTER_COPY:-}" ] && bash -c "$TEST_HOOK_AFTER_COPY"
( cd "$OUT/code" && sha256sum * > "$OUT/provenance/code.sha256" )
T=$OUT/code
V=$OUT/provenance
AV=$V/approved            # 이후 검증·대조·집계는 이 복사본 목록만 쓴다
# 1-1. 실행 사본을 승인 목록 복사본으로 검증
if [ -f "$AV/code.sha256" ]; then
  ( cd "$OUT/code" && sha256sum -c --quiet "$AV/code.sha256" ) || die "execution copies differ from approved list (code)"
  if [ -f "$AV/design.sha256" ] && [ -f "$OUT/code/$(basename "$DESIGN")" ]; then
    ( cd "$OUT/code" && sha256sum -c --quiet "$AV/design.sha256" ) || die "execution copy of design differs from approved list"
  fi
  echo "copies_verified_against $AV" >> "$V/copy_check.txt"
elif [ "$MODE" = formal ]; then die "formal: approved list copy missing"
else echo "copies_not_verified (smoke without approved list)" >> "$V/copy_check.txt"; fi
if [ "$MODE" = formal ]; then
  [ "$(sha256sum < "$OUT/jobs.tsv" | cut -d' ' -f1)" = "$(awk '$2=="jobs.tsv"{print $1}' "$AV/SHA256SUMS")" ] || die "jobs.tsv differs from approved list copy"
  grep -v ' jobs.tsv$' "$AV/SHA256SUMS" | sha256sum -c --quiet - || die "ETF differs from approved list copy"
  sha256sum -c --quiet "$AV/stage2_sources.sha256" || die "stage2 sources differ from approved list copy"
fi
if [ "${STOP_AFTER_COPY_VERIFY:-0}" = 1 ]; then echo "copy verify ok; stopping (test)"; exit 0; fi
{
  echo "mode $MODE"; echo "started_utc $(date -u +%FT%TZ)"
  echo "repo_head $(git -C $R/repo rev-parse HEAD)"; echo "repo_dirty_files $(git -C $R/repo status --porcelain | wc -l)"
  echo "wt_head $(git -C $W rev-parse HEAD)"; echo "wt_dirty_files $(git -C $W status --porcelain | wc -l)"
  $PY -c "import sys,pandas,numpy,yaml;print('python',sys.version.split()[0],'pandas',pandas.__version__,'numpy',numpy.__version__,'pyyaml',yaml.__version__)"
  echo "universe_mode strict_pit"; echo "parallel $P"
} > "$V/env.txt"
git -C $R/repo status --porcelain > "$V/repo_status.txt"; git -C $W status --porcelain > "$V/wt_status.txt"
input_hashes() {
  ( cd $W && sha256sum config/config.yaml results/wfo_run_20260324_125326/final_candidate_manifest.json | sed 's#config/config.yaml#config.yaml(content hash only)#' )
  sha256sum "$(realpath $ETF)" "$OUT/jobs.tsv"
}
input_hashes > "$V/inputs.sha256"
# 2. DB 체크섬(기준: 수집 복구 준비 때 원본 DB 체크섬, FLOAT 표 재적재 뒤 값)
if [ "${SKIP_DB_CHECKSUM:-0}" != 1 ]; then
  db_checksums() { for t in CalculatedIndicators CompanyInfo DailyStockPrice DailyStockTier FinancialData InvestorTradingTrend MarketCapDaily TickerUniverseHistory TickerUniverseSnapshot; do
    echo "$t $(mysql -uroot stocks -N -e "SELECT COUNT(*) FROM $t") $(mysql -uroot stocks -N -e "CHECKSUM TABLE $t" | awk '{print $2}')"; done; }
  db_checksums > "$V/db_checksum.txt"
  mysql -uroot stocks -N -e "SELECT MAX(date) FROM DailyStockPrice" > "$V/db_max_date.txt"
  REF=$R/transfer/pre-collect-20261122/checksums_original.tsv; echo "db_reference $REF" >> "$V/env.txt"
  awk 'NR==FNR {want[$1]=1; next} ($1 in want) {print $1, $2, $3}' "$V/db_checksum.txt" "$REF" | sort > "$V/db_reference.txt"
  diff <(sort "$V/db_checksum.txt") "$V/db_reference.txt" > "$V/db_diff.txt" || die "DB checksum differs from reference (see $V/db_diff.txt)"
  echo "db_matches_reference yes" >> "$V/env.txt"
else
  echo "db_checksum skipped (smoke)" >> "$V/env.txt"
fi
run_one() {
  local kind=$1 k=$2 win=$3 s=$4 e=$5 tax=$6 div=$7 pre=$8 t0 rc
  t0=$(date +%s)
  ( cd $W && MAGICSPLIT_UNIVERSE_MODE=strict_pit $PY -B $T/continuous_account_run.py "$s" "$e" "$OUT/runs/$pre" --tax "$tax" --div "$div" --hold none ) > "$OUT/runs/$pre.log" 2>&1 && rc=0 || rc=$?
  printf '%s\t%s\t%s\n' "$pre" "$rc" "$(( $(date +%s) - t0 ))" >> "$OUT/job_status.tsv"
}
export -f run_one; export OUT PY T W
printf 'prefix\texit\tseconds\n' > "$OUT/job_status.tsv"
# 3. 3월 재현 먼저
if awk -F'\t' 'NR>1 && $1=="anchor" {f=1} END {exit !f}' "$OUT/jobs.tsv"; then
  awk -F'\t' 'NR>1 && $1=="anchor"' "$OUT/jobs.tsv" | xargs -L1 bash -c 'run_one "$@"' _
  ( cd $W && $PY -B $T/continuous_account_verdict.py "$OUT" $ETF --mode $MODE --anchor-only ) > "$OUT/anchor_check.json" || die "anchor reproduction failed (see anchor_check.json)"
elif [ "$MODE" = formal ]; then die "formal: anchor job missing"; fi
# 4. 나머지
awk -F'\t' 'NR>1 && $1!="anchor"' "$OUT/jobs.tsv" | xargs -P "$P" -L1 bash -c 'run_one "$@"' _ || true
echo "jobs_finished_utc $(date -u +%FT%TZ)" >> "$V/env.txt"
EXPECTED=$(( $(wc -l < "$OUT/jobs.tsv") - 1 ))
awk -F'\t' -v n=$EXPECTED 'NR>1 {c++; if ($2!=0) bad=1} END {exit (bad || c!=n)}' "$OUT/job_status.tsv" || die "some jobs failed or missing (see job_status.tsv); no verdict"
# 4-1. 실행 중 입력 변경 확인(설정·manifest·ETF·작업표 해시, DB 체크섬)
input_hashes | diff - "$V/inputs.sha256" > "$V/inputs_recheck.diff" || die "inputs changed during the run (see inputs_recheck.diff)"
if [ "${SKIP_DB_CHECKSUM:-0}" != 1 ]; then
  db_checksums | diff - "$V/db_checksum.txt" > "$V/db_recheck.diff" || die "DB changed during the run (see db_recheck.diff)"
  echo "db_recheck_after_jobs yes" >> "$V/env.txt"
fi
# 5. 2단계 대조(판정 전; 실패해도 직접 비교는 보존되고 그 기준의 2단계 해석만 유보) → 필수 검사·집계·판정(실패 시 판정 미발행)
S2RC=0
( cd $W && $PY -B $T/continuous_account_stage2_check.py "$OUT" $STAGE2 $ETF --mode $MODE --approved "$AV/stage2_sources.sha256" ) > "$OUT/stage2_check.log" 2>&1 || S2RC=$?
echo "$S2RC" > "$OUT/stage2_check.exit"     # 비0이면 판정기가 두 기준 모두 2단계 해석을 withheld로 둔다
[ "$S2RC" = 0 ] || echo "stage2 reconcile exited $S2RC (see stage2_check.log); stage2 interpretation will be withheld" >&2
( cd $W && $PY -B $T/continuous_account_verdict.py "$OUT" $ETF --mode $MODE --approved-dir "$AV" ) > "$OUT/verdict.log" 2>&1 || die "checks failed; no verdict (see verdict.log, checks.json)"
echo "finished_utc $(date -u +%FT%TZ)" >> "$V/env.txt"
# 6. 최종 산출물 해시
( cd "$OUT" && find . -type f ! -name SHA256SUMS ! -name '.opened' | sort | xargs sha256sum > SHA256SUMS )
echo "done: $OUT"
