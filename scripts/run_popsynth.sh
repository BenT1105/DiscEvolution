#!/bin/bash
#
# run_popsynth_student.sh
# ------------------------
# Launch a grid of DiscEvolution runs (run_model_popsynth.py) over
# (M, Mdot, Rd, pla_eff, f_plt, alpha), running up to $NPROC of them at a time.
#
# This is a script to sweep a parameter grid, run in parallel,
# be safe to re-launch after an interruption and skip any combination
# whose output file exists and is marked complete. So this script
# can just launch everything every time; the Python side figures
# out what's actually left to do.
#
# Usage:
#   ./run_popsynth_popsynth.sh
#
# To run this fully in the background, detached from your terminal (so it
# keeps going after you close your laptop or log out of an ssh session):
#   nohup setsid ./run_popsynth_student.sh > master.log 2>&1 &

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Parameter grid. Edit these six lines to change what gets run.
# ---------------------------------------------------------------------------

M_VALUES="0.05 0.075 0.1 0.125 0.15"
MDOT_VALUES="1e-9 3e-9 1e-8 3e-8 1e-7 3e-7"
RD_VALUES="50 100 150 200"
PLA_EFF_VALUES="0.1 0.3 0.5 0.7 0.9"
F_PLT_VALUES="0.1 0.3 0.5 0.7 0.9"
ALPHA_VALUES="1e-5 3e-5 1e-4 3e-4 1e-3"

# ---------------------------------------------------------------------------
# 2. Run name, config file, where output/figures/logs go, and how many runs
#    at once. Output and figure files are written under ./$RUN_NAME/.
# ---------------------------------------------------------------------------

RUN_NAME="popsynth"
PLOT=false

CONFIG_NAME="popsynth_default_config.json"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config/$CONFIG_NAME"
OUTDIR="${DISCEVOLUTION_OUTPUT:-$SCRIPT_DIR/$RUN_NAME/output}"
FIGDIR="${DISCEVOLUTION_FIGURE_DIR:-$SCRIPT_DIR/$RUN_NAME/figure}"
LOGDIR="$SCRIPT_DIR/logs"
MASTER_LOG="$LOGDIR/${RUN_NAME}_log.csv"
NPROC=8

mkdir -p "$LOGDIR" "$OUTDIR" "$FIGDIR"

# One master CSV log for the whole sweep (separate from the per-run .out/.err
# files also in $LOGDIR): a row per run mapping its output filename back to
# the (M, Mdot, Rd, pla_eff, f_plt, alpha) that produced it. Truncated fresh
# each launch, since this script always relaunches the full grid anyway.
echo "filename,M,Mdot,Rd,pla_eff,f_plt,alpha" > "$MASTER_LOG"

N_TOTAL=$(( $(wc -w <<< "$M_VALUES") * $(wc -w <<< "$MDOT_VALUES") * $(wc -w <<< "$RD_VALUES") \
          * $(wc -w <<< "$PLA_EFF_VALUES") * $(wc -w <<< "$F_PLT_VALUES") * $(wc -w <<< "$ALPHA_VALUES") ))

echo "Config:     $CONFIG_FILE"
echo "Run name:   $RUN_NAME"
echo "Output to:  $OUTDIR"
echo "Plot:       $PLOT"
echo "Figures to: $FIGDIR"
echo "Logs to:    $LOGDIR"
echo "Master log: $MASTER_LOG"
echo "Grid size:  $N_TOTAL combinations"
echo

# ---------------------------------------------------------------------------
# 3. One job per (M, Mdot, Rd, pla_eff, f_plt, alpha) combination.
# ---------------------------------------------------------------------------

# Position (0-based) of $1 within the space-separated list $2, or -1 if absent.
index_of() {
    local value="$1" list="$2" i=0 v
    for v in $list; do
        [[ "$v" == "$value" ]] && { echo "$i"; return; }
        ((i++))
    done
    echo "-1"
}

run_one() {

    local M="$1" Mdot="$2" Rd="$3" pla_eff="$4" f_plt="$5" alpha="$6"

    local i j k x y z
    i=$(index_of "$M" "$M_VALUES")
    j=$(index_of "$Mdot" "$MDOT_VALUES")
    k=$(index_of "$Rd" "$RD_VALUES")
    x=$(index_of "$pla_eff" "$PLA_EFF_VALUES")
    y=$(index_of "$f_plt" "$F_PLT_VALUES")
    z=$(index_of "$alpha" "$ALPHA_VALUES")

    local tag="${RUN_NAME}_${i}_${j}_${k}_${x}_${y}_${z}"

    local extra_args=(--output_dir "$OUTDIR" --output_filename "$tag" --figure_dir "$FIGDIR")
    [[ "$PLOT" == true ]] && extra_args+=(--plot)

    echo "[$(date +%T)] Launching $tag (M=$M Mdot=$Mdot Rd=$Rd pla_eff=$pla_eff f_plt=$f_plt alpha=$alpha; skips itself if already done -- see .out log)"
    python3 "$SCRIPT_DIR/run_model_popsynth.py" --config "$CONFIG_FILE" \
        --M "$M" --Mdot "$Mdot" --Rd "$Rd" --pla_eff "$pla_eff" --f_plt "$f_plt" --alpha "$alpha" \
        "${extra_args[@]}" \
        > "$LOGDIR/${tag}.out" 2> "$LOGDIR/${tag}.err"

    printf '%s,%s,%s,%s,%s,%s,%s\n' "${tag}.h5" "$M" "$Mdot" "$Rd" "$pla_eff" "$f_plt" "$alpha" >> "$MASTER_LOG"
}

export -f index_of run_one
export SCRIPT_DIR CONFIG_FILE OUTDIR FIGDIR LOGDIR MASTER_LOG RUN_NAME PLOT
export M_VALUES MDOT_VALUES RD_VALUES PLA_EFF_VALUES F_PLT_VALUES ALPHA_VALUES

if command -v parallel >/dev/null 2>&1; then
    parallel -j "$NPROC" run_one {1} {2} {3} {4} {5} {6} \
        ::: $M_VALUES ::: $MDOT_VALUES ::: $RD_VALUES ::: $PLA_EFF_VALUES ::: $F_PLT_VALUES ::: $ALPHA_VALUES

else
    # Fallback if GNU parallel isn't installed: a plain bash job-control
    # loop that does the same thing (launch in the background, cap how
    # many run at once with `wait`).
    echo "(GNU parallel not found -- using a plain bash loop instead)"
    count=0
    for M in $M_VALUES; do
      for Mdot in $MDOT_VALUES; do
        for Rd in $RD_VALUES; do
          for pla_eff in $PLA_EFF_VALUES; do
            for f_plt in $F_PLT_VALUES; do
              for alpha in $ALPHA_VALUES; do
                run_one "$M" "$Mdot" "$Rd" "$pla_eff" "$f_plt" "$alpha" &
                ((count++))
                if ((count % NPROC == 0)); then wait; fi
              done
            done
          done
        done
      done
    done
    wait
fi

echo
echo "[$(date +%T)] All simulations complete."
