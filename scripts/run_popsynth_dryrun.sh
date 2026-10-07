#!/bin/bash
#
# run_popsynth_dryrun.sh
# ----------------------
# Write out the run_model_popsynth.py commands that run_popsynth.sh would
# launch, one per line, without running any of them. The only file written is
# the listing itself, ~/simulations/dryruns/${RUN_NAME}.txt (overwritten each
# time); no output, figure or log directories are created, and the master
# CSV log is untouched.
#
# The parameter grid and run settings are read straight out of
# run_popsynth.sh, so this always reflects whatever that script currently
# contains; there is nothing to edit here.
#
# Usage:
#   ./run_popsynth_dryrun.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_SCRIPT="$SCRIPT_DIR/run_popsynth.sh"

if [[ ! -f "$SWEEP_SCRIPT" ]]; then
    echo "ERROR: $SWEEP_SCRIPT not found" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Pull the grid + run settings (sections 1 and 2) from run_popsynth.sh.
# ---------------------------------------------------------------------------

# These are plain variable assignments, evaluated here in the order they
# appear in run_popsynth.sh (the paths build on RUN_NAME, CONFIG_NAME and
# SCRIPT_DIR, which is the same directory for both scripts).
eval "$(grep -E '^(M_VALUES|MDOT_VALUES|RD_VALUES|PLA_EFF_VALUES|F_PLT_VALUES|ALPHA_VALUES|RUN_NAME|PLOT|CONFIG_NAME|CONFIG_FILE|OUTDIR|FIGDIR|LOGDIR|MASTER_LOG|NPROC|RUN_LOG)=' "$SWEEP_SCRIPT")"

N_TOTAL=$(( $(wc -w <<< "$M_VALUES") * $(wc -w <<< "$MDOT_VALUES") * $(wc -w <<< "$RD_VALUES") \
          * $(wc -w <<< "$PLA_EFF_VALUES") * $(wc -w <<< "$F_PLT_VALUES") * $(wc -w <<< "$ALPHA_VALUES") ))

# Everything from here on (the summary, then the commands) goes to the
# dry-run listing instead of the terminal.
DRYRUN_FILE="$HOME/simulations/dryruns/${RUN_NAME}.txt"
mkdir -p "$(dirname "$DRYRUN_FILE")"
echo "Writing dry run to: $DRYRUN_FILE"
exec > "$DRYRUN_FILE"

{
    echo "DRY RUN -- nothing is run"
    echo "Config:     $CONFIG_FILE"
    echo "Run name:   $RUN_NAME"
    echo "Output to:  $OUTDIR"
    echo "Plot:       $PLOT"
    echo "Figures to: $FIGDIR"
    echo "Logs to:    $LOGDIR"
    echo "Master log: $RUN_LOG"
    echo "CSV log:    $MASTER_LOG"
    echo "Grid size:  $N_TOTAL combinations ($NPROC at a time)"
    echo
}

# ---------------------------------------------------------------------------
# 2. One command per (M, Mdot, Rd, pla_eff, f_plt, alpha) combination, in the
#    same order and with the same index-based tag as run_popsynth.sh.
# ---------------------------------------------------------------------------

i=0
for M in $M_VALUES; do
  j=0
  for Mdot in $MDOT_VALUES; do
    k=0
    for Rd in $RD_VALUES; do
      x=0
      for pla_eff in $PLA_EFF_VALUES; do
        y=0
        for f_plt in $F_PLT_VALUES; do
          z=0
          for alpha in $ALPHA_VALUES; do
            tag="${RUN_NAME}_${i}_${j}_${k}_${x}_${y}_${z}"

            cmd=(python3 "$SCRIPT_DIR/run_model_popsynth.py" --config "$CONFIG_FILE"
                 --M "$M" --Mdot "$Mdot" --Rd "$Rd" --pla_eff "$pla_eff" --f_plt "$f_plt" --alpha "$alpha"
                 --output_dir "$OUTDIR" --output_filename "$tag" --figure_dir "$FIGDIR")
            [[ "$PLOT" == true ]] && cmd+=(--plot)

            # %q quotes each word so the printed line can be pasted into a shell as-is
            printf '%q ' "${cmd[@]}"
            printf '> %q 2> %q\n' "$LOGDIR/${tag}.out" "$LOGDIR/${tag}.err"

            z=$((z + 1))
          done
          y=$((y + 1))
        done
        x=$((x + 1))
      done
      k=$((k + 1))
    done
    j=$((j + 1))
  done
  i=$((i + 1))
done
