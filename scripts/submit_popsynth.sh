#!/bin/bash
#
# submit_popsynth.sh
# ------------------
# Slurm wrapper for run_popsynth.sh on Nibi. Submit from the folder that
# contains both scripts:
#   sbatch submit_popsynth.sh
#
#SBATCH --account=def-mbalogh
#SBATCH --job-name=popsynth
#SBATCH --time=5-00:00              # whole grid, not one run (max 7-00:00)
#SBATCH --cpus-per-task=160         # runs at once (sets NPROC)
#SBATCH --mem-per-cpu=2G            # memory per run
#SBATCH --output=/scratch/bntobin/simulations/%x-%j.out
#SBATCH --mail-user=bntobin@uwaterloo.ca
#SBATCH --mail-type=END,FAIL

module load python/3.12
source ~/envs/DiscEvo_venv/bin/activate
export OMP_NUM_THREADS=1            # one thread per run
export MPLBACKEND=Agg               # plotting without a display
export SIMDIR=~/scratch/simulations # output location on Nibi

bash "$SLURM_SUBMIT_DIR/run_popsynth.sh"
