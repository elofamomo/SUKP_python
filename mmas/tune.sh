#!/usr/bin/env bash
# Tunes MMAS: runs mmas_run once per configuration on the tuning instances and ranks the
# configurations by the average gap of the mean profit to the best known value.
#
# Usage (from the repo root, after building mmas/build):
#   mmas/tune.sh CONFIGS OUT_DIR [RUNS]
# CONFIGS has one configuration per line: a name, then mmas_run options, e.g.
#   static_b2_r0.1   --heuristic static --beta 2 --rho 0.1
# Lines starting with # are ignored. Every configuration uses the same seeds (0..RUNS-1), and a
# configuration whose CSV already exists in OUT_DIR is skipped, so an interrupted tuning resumes.
# OUT_DIR/ranking.csv lists the configurations, lowest average gap of the mean first.
# Environment: THREADS (default 6), INSTANCES (default: the tuning subset below).
set -euo pipefail
export LC_ALL=C  # "." decimals in awk and sort, whatever the desktop locale

# Tuning subset: 8 instances covering sizes 185-800, m < n, m = n, m > n and both densities.
# The other 52 instances are not used for tuning, so results on them show whether the tuned
# parameters generalize.
INSTANCES=${INSTANCES:-"sukp_185_200_0.15_0.85 sukp_200_200_0.10_0.75 sukp_285_300_0.15_0.85
sukp_300_285_0.10_0.75 sukp_400_400_0.15_0.85 sukp_500_485_0.10_0.75 sukp_600_585_0.10_0.75
sukp_785_800_0.15_0.85"}

if [[ $# -lt 2 ]]; then
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
configs=$1
out=$2
runs=${3:-20}
threads=${THREADS:-6}
mkdir -p "$out"

while read -r name opts; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    [[ -f "$out/$name.csv" ]] && continue
    start=$(date +%s)
    # shellcheck disable=SC2086  # $INSTANCES and $opts hold several words
    ./mmas/build/mmas_run --instances $INSTANCES --runs "$runs" --threads "$threads" $opts \
        --out "$out/$name.csv" > "$out/$name.log" < /dev/null
    echo "$name: $(grep '^average over' "$out/$name.log")  [$(( $(date +%s) - start ))s]"
done < "$configs"

# Ranking over every configuration of CONFIGS found in OUT_DIR
{
    echo "config,gap_mean_%,gap_best_%,bks_hits,time_s,options"
    while read -r name opts; do
        [[ -z "$name" || "$name" == \#* || ! -f "$out/$name.csv" ]] && continue
        awk -F, -v name="$name" -v opts="$opts" '
            NR > 1 { gm += $9; gb += $8; t += $5; n++; if ($8 + 0 == 0) h++ }
            END { printf "%s,%.3f,%.3f,%d,%.3f,\"%s\"\n", name, gm / n, gb / n, h, t / n, opts }' "$out/$name.csv"
    done < "$configs" | sort -t, -k2,2g
} > "$out/ranking.csv"
echo
echo "top 15 of $(($(wc -l < "$out/ranking.csv") - 1)) configurations ($out/ranking.csv):"
head -16 "$out/ranking.csv" | column -s, -t
