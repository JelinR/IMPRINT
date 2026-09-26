#!/bin/bash
#SBATCH --job-name=mdetect_5_2
#SBATCH --partition=testing
#SBATCH --output=scripts/logs/mdetect_5_2.out
#SBATCH --error=scripts/logs/mdetect_5_2.err
#SBATCH --account=phd
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00

set -euo pipefail

#Create Log Directory
RUN_DIR="/home/akkara/OneMap/scripts/logs/onemap/mdetect_5_2"

mkdir -p "$RUN_DIR"
echo "Logs for this run → $RUN_DIR"

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

CURR_PYTHON="/home/akkara/.conda/envs/one_map/bin/python"

RUN_ARGS="--config config/mon/eval_conf.yaml  \
            --PlanningConf.using_ov \
            --EvalConf.results_dir /home/akkara/OneMap/results/RAL/ovon/equal/syn/mdetect_5/split_2 \
            --EvalConf.run_split 2 \
            --PlanningConf.multi_detect \
            --PlanningConf.yolo_confidence 0.9 \
            --PlanningConf.scrape_num 5"

# -------------------------
# Run experiment
# -------------------------
echo "Current directory: $(pwd)"
echo "Python path: $(which python)"

exec >> "$RUN_DIR/experiment.log" 2>&1
$CURR_PYTHON eval_habitat.py $RUN_ARGS


# -------------------------
# Cleanup
# -------------------------
echo "Experiment finished, shutting down servers..."
echo "Task completed"

