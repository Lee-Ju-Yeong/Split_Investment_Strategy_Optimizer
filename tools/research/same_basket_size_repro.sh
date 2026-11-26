#!/bin/bash
# 주문율 격자(same-basket-size-r1) 2.2% 재현 대조(설계 5절 "2.2%에서 r1 X와 값 일치", H·K 재사용 값 일치).
# usage: bash tools/research/same_basket_size_repro.sh OUT_DIR [R1_PREFIX ...]      (OUT_DIR은 새 폴더)
#   R1_PREFIX마다(기본 3개) r1 정식 작업과 같은 가정·기간으로
#     X: same_basket_size_run.py --ratio 0.022 --r1 (기록기를 붙인 채 r1 X 파일 11개 바이트 일치)
#     H·K: same_basket_ledger.py(r1 그대로) → r1의 _H_*·_K_* 파일 8개 바이트 일치
#   를 동시에 돌린다. 결과는 OUT_DIR/repro_check.json(통과 여부, 파일별 일치, 작업별 걸린 시간). 성과 값은 출력하지 않는다.
#   하위 출력은 성공하면 지우고 실패하면 마지막 40줄만 .err로 남긴다("final": 줄 제거). DB는 SELECT만 한다.
set -uo pipefail
R=/mnt/volume/workspace/split-research
PY=$R/env/bin/python; W=$R/wt-79051b5; T=$(dirname "$(realpath "$0")"); ETF=$R/data/etf/069500.csv
R1=$R/runs/same-basket-r1/full
OUT=$(realpath -m "$1"); shift
[ -e "$OUT" ] && { echo "ERROR: $OUT already exists" >&2; exit 1; }
PREFIXES=("$@"); [ ${#PREFIXES[@]} -gt 0 ] || PREFIXES=(main_b2014_k0_approx_freeze main_b2022_k20_upper_zero main_b2018_k5_none_freeze)
mkdir -p "$OUT/runs"
one() {
  local pre=$1 row s e y d z t0 rc ok
  row=$(awk -F'\t' -v p="$pre" '$11==p' "$R1/jobs.tsv"); [ -n "$row" ] || { echo "{\"prefix\": \"$pre\", \"passed\": false, \"error\": \"not in r1 jobs\"}"; return; }
  y=$(cut -f3 <<<"$row"); d=$(cut -f5 <<<"$row"); z=$(cut -f6 <<<"$row"); s=$(cut -f9 <<<"$row"); e=$(cut -f10 <<<"$row")
  t0=$(date +%s)
  ( cd $W && export MAGICSPLIT_UNIVERSE_MODE=strict_pit && \
    $PY -B "$T/same_basket_size_run.py" "$s" "$e" "$OUT/runs/$pre" --basket "$R1/baskets/basket_$y.csv" --ratio 0.022 --div "$d" --delist "$z" --r1 "$R1/runs/$pre" && \
    $PY -B "$T/same_basket_ledger.py" "$s" "$e" "$OUT/runs/$pre" --basket "$R1/baskets/basket_$y.csv" --div "$d" --delist "$z" --etf "$ETF" ) > "$OUT/runs/$pre.out" 2>&1
  rc=$?
  [ "$rc" = 0 ] || tail -n 40 "$OUT/runs/$pre.out" | grep -v '"final":' > "$OUT/runs/$pre.err"
  rm -f "$OUT/runs/$pre.out"
  local hk="" same=1
  for f in H_curve.csv H_pos.csv H_buys.csv H_divs.csv H_divskips.csv H_meta.json K_curve.csv K_meta.json; do
    if cmp -s "$OUT/runs/${pre}_$f" "$R1/runs/${pre}_$f"; then ok=true; else ok=false; same=0; fi; hk="$hk\"$f\": $ok, "
  done
  local xok=false; grep -q '"passed": true' "$OUT/runs/${pre}_r1cmp.json" 2>/dev/null && xok=true
  local passed=false; [ "$rc" = 0 ] && [ "$same" = 1 ] && [ "$xok" = true ] && passed=true
  echo "{\"prefix\": \"$pre\", \"passed\": $passed, \"exit\": $rc, \"seconds\": $(( $(date +%s) - t0 )), \"x_r1cmp_passed\": $xok, \"hk\": {${hk%, }}}"
}
export -f one; export R PY W T ETF R1 OUT
t0=$(date +%s)
printf '%s\n' "${PREFIXES[@]}" | xargs -P ${#PREFIXES[@]} -I{} bash -c 'one "$@" > "$OUT/runs/$1.res"' _ {}
all=true; items=""
for p in "${PREFIXES[@]}"; do r=$(cat "$OUT/runs/$p.res"); grep -q '"passed": true' <<<"$r" || all=false; items="$items$r, "; rm -f "$OUT/runs/$p.res"; done
printf '{"passed": %s, "parallel": %s, "seconds": %s, "jobs": [%s]}\n' "$all" "${#PREFIXES[@]}" "$(( $(date +%s) - t0 ))" "${items%, }" > "$OUT/repro_check.json"
echo "repro passed=$all"
[ "$all" = true ]
