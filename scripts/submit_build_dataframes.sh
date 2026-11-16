#!/bin/bash
#
# submit_build_dataframes.sh
# --------------------------
# Slurm wrapper for build_dataframes.py on Nibi. Submit from the folder 
# that contains both scripts:
#   sbatch submit_build_dataframes.sh
#
#SBATCH --account=def-mbalogh
#SBATCH --job-name=build_dataframes
#SBATCH --time=0-03:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --output=/scratch/bntobin/jobs/%x-%j.out
#SBATCH --mail-user=bntobin@uwaterloo.ca
#SBATCH --mail-type=END,FAIL

RUN_NAME="popsynth_2"
SIMDIR=~/scratch/simulations
REBUILD_ARGS=""

module load python/3.12
source ~/envs/DiscEvo_venv/bin/activate
export OMP_NUM_THREADS=1

python "$SLURM_SUBMIT_DIR/build_dataframes.py" \
    --model "$RUN_NAME" \
    --output_dir "$SIMDIR/$RUN_NAME/output" \
    --save_dir "$SIMDIR/$RUN_NAME/dataframes" \
    $REBUILD_ARGS
