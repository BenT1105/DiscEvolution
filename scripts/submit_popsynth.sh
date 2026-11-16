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
#SBATCH --time=5-00:00
#SBATCH --cpus-per-task=160
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/scratch/bntobin/jobs/%x-%j.out
#SBATCH --mail-user=bntobin@uwaterloo.ca
#SBATCH --mail-type=END,FAIL

module load python/3.12
source ~/envs/DiscEvo_venv/bin/activate
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
export SIMDIR=~/scratch/simulations

bash "$SLURM_SUBMIT_DIR/run_popsynth_nibi.sh"
