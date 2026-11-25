#!/bin/bash
# 같은 종목 대조(same-basket-r1) 재현 대조(설계 10절 1): 명단 제한을 끈 실행이 기존 연속 계좌 산출물과 체결·현금까지 같은지 본다.
# usage: bash tools/research/same_basket_repro.sh OUT_DIR      (OUT_DIR은 새 폴더)
#   same_basket_run.py를 --no-restrict --eligible-all(준비일 없음, 매일 후보 기록)로 2014-01-02~2015-12-30, 세율 period·배당 approx로 돌려
#   runs/continuous-account-r1/full/runs/restart_k0_20140101_period_approx 의 산출물 8개와 바이트 단위로 비교한다.
#   결과는 OUT_DIR/repro_check.json(passed, 파일별 일치, 걸린 시간). 성과 값은 출력하지 않는다. DB는 SELECT만 한다.
#   실행기 출력은 성공하면 지우고, 실패하면 마지막 40줄만 repro.err로 남긴다("final": 필드가 든 줄은 버림).
set -uo pipefail
R=/mnt/volume/workspace/split-research
PY=$R/env/bin/python; W=$R/wt-79051b5; T=$(dirname "$(realpath "$0")")
REF=$R/runs/continuous-account-r1/full/runs/restart_k0_20140101_period_approx
BASKET=${BASKET_DIR:-$R/runs/same-basket-r1/baskets}/basket_2014.csv
OUT=$(realpath -m "$1")
[ -e "$OUT" ] && { echo "ERROR: $OUT already exists" >&2; exit 1; }
mkdir -p "$OUT"
t0=$(date +%s)
( cd $W && MAGICSPLIT_UNIVERSE_MODE=strict_pit $PY -B "$T/same_basket_run.py" 2014-01-02 2015-12-30 "$OUT/repro" --basket "$BASKET" --no-restrict --eligible-all \
    --tax period --div approx --hold none ) > "$OUT/repro.out" 2>&1
rc=$?; secs=$(( $(date +%s) - t0 ))
[ "$rc" = 0 ] || tail -n 40 "$OUT/repro.out" | grep -v '"final":' > "$OUT/repro.err"
rm -f "$OUT/repro.out"
same=1; files=""
for s in curve fills divs state endpos unfilled addsignals holds; do
  if cmp -s "$OUT/repro_$s.csv" "${REF}_$s.csv"; then ok=true; else ok=false; same=0; fi
  files="$files\"$s\": $ok, "
done
[ "$rc" = 0 ] || same=0
passed=false; [ "$same" = 1 ] && passed=true
printf '{"passed": %s, "exit": %s, "seconds": %s, "files": {%s}, "reference": "%s"}\n' "$passed" "$rc" "$secs" "${files%, }" "$REF" > "$OUT/repro_check.json"
echo "repro passed=$passed exit=$rc seconds=$secs"
[ "$passed" = true ]
