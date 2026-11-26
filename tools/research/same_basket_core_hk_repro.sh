#!/bin/bash
# same-basket-r2: H·K 재현 대조. r1 H 장부를 r2 장부 실행기(same_basket_core_ledger.py --hk-only)로 다시 돌린 H·K 산출물 8종이
# r1 본실행(runs/same-basket-r1/full)의 같은 이름 파일과 바이트 단위로 같은지 본다. r1 파일은 열지 않고 r1 SHA256SUMS의 해시와만 비교한다.
# usage: bash tools/research/same_basket_core_hk_repro.sh OUT_DIR [PARALLEL] [ID_LIST]
#   OUT_DIR은 새 폴더. 작업은 r1 full/jobs.tsv의 main 108행(ID_LIST를 주면 그 id만, 쉼표 구분).
#   결과는 OUT_DIR/hk_repro.json(passed, 작업 수, 파일 일치 수, 다른 파일 이름, 걸린 시간). 성과 값은 출력하지 않는다. DB는 SELECT만 한다.
#   하위 출력은 성공하면 지우고, 실패하면 마지막 40줄만 남긴다.
set -uo pipefail
R=/mnt/volume/workspace/split-research
PY=$R/env/bin/python; W=$R/wt-79051b5; T=$(dirname "$(realpath "$0")"); ETF=$R/data/etf/069500.csv
R1=$R/runs/same-basket-r1/full; BASKETS=${BASKET_DIR:-$R/runs/same-basket-r1/baskets}
OUT=$(realpath -m "$1"); P=${2:-3}; IDS=${3:-}
[ -e "$OUT" ] && { echo "ERROR: $OUT already exists" >&2; exit 1; }
[[ "$P" =~ ^[1-6]$ ]] || { echo "ERROR: PARALLEL must be 1-6" >&2; exit 1; }
mkdir -p "$OUT/runs"
t0=$(date +%s)
one() {
  local id=$1 kind=$2 basket=$3 k=$4 div=$5 delist=$6 s=$9 e=${10} pre=${11} rc
  ( cd $W && MAGICSPLIT_UNIVERSE_MODE=strict_pit $PY -B "$T/same_basket_core_ledger.py" "$s" "$e" "$OUT/runs/$pre" --basket "$BASKETS/basket_$basket.csv" \
      --div "$div" --delist "$delist" --kind main --hk-only --etf "$ETF" ) > "$OUT/runs/$pre.out" 2>&1 && rc=0 || rc=$?
  [ "$rc" = 0 ] || tail -n 40 "$OUT/runs/$pre.out" | grep -v '"final":' > "$OUT/runs/$pre.err"
  rm -f "$OUT/runs/$pre.out"
  echo "$pre $rc" >> "$OUT/status.txt"
}
export -f one; export OUT PY T W ETF BASKETS
awk -F'\t' -v ids=",$IDS," 'NR>1 && $2=="main" && (ids==",," || index(ids, ","$1",")>0)' "$R1/jobs.tsv" | xargs -r -P "$P" -L1 bash -c 'one "$@"' _
touch "$OUT/status.txt"
n=0; same=0; bad=""; failed=0
while read -r pre rc; do
  [ "$rc" = 0 ] || failed=$((failed+1))
  for f in H_curve.csv H_pos.csv H_buys.csv H_divs.csv H_divskips.csv H_meta.json K_curve.csv K_meta.json; do
    n=$((n+1)); h=$(sha256sum "$OUT/runs/${pre}_$f" 2>/dev/null | cut -d' ' -f1)
    if [ -n "$h" ] && grep -qx "$h  ./runs/${pre}_$f" "$R1/SHA256SUMS"; then same=$((same+1)); else bad="$bad ${pre}_$f"; fi
  done
done < <(sort "$OUT/status.txt")
jobs=$(wc -l < "$OUT/status.txt"); passed=false
[ "$failed" = 0 ] && [ "$same" = "$n" ] && [ "$n" -gt 0 ] && passed=true
printf '{"passed": %s, "jobs": %s, "failed_jobs": %s, "files": %s, "same": %s, "differ": "%s", "seconds": %s, "reference": "%s"}\n' \
  "$passed" "$jobs" "$failed" "$n" "$same" "${bad# }" "$(( $(date +%s) - t0 ))" "$R1/SHA256SUMS" > "$OUT/hk_repro.json"
echo "hk_repro passed=$passed jobs=$jobs files=$n same=$same"
[ "$passed" = true ]
