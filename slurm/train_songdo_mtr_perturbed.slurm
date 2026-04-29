#!/bin/bash
#SBATCH --job-name=mtr_songdo_perturbed
#SBATCH --output=/home/%u/projects/Songdo/logs/%x_%j.out
#SBATCH --error=/home/%u/projects/Songdo/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --account=master

set -euo pipefail

WORKROOT=${WORKROOT:-/home/$USER/projects/Songdo}
LOG_DIR=${LOG_DIR:-$WORKROOT/logs}
UNITRAJ_ROOT=${UNITRAJ_ROOT:-$WORKROOT/UniTraj_Songdo}
TRAIN_DATA=${TRAIN_DATA:-/scratch/izar/$USER/songdo_drive/train}
VAL_DATA=${VAL_DATA:-/scratch/izar/$USER/songdo_drive/test}
CACHE_ROOT=${CACHE_ROOT:-/scratch/izar/$USER/songdo_cache/mtr_perturbed}
VENV_PATH=${VENV_PATH:-/home/$USER/venvs/mtr-izar/bin/activate}
EXP_NAME=${EXP_NAME:-mtr_songdo_perturbed}
LOAD_NUM_WORKERS=${LOAD_NUM_WORKERS:-4}
MAX_EPOCHS=${MAX_EPOCHS:-40}

mkdir -p "$LOG_DIR"

module purge
module load gcc/11.3.0
module load python

source "$VENV_PATH"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

for required_dir in "$UNITRAJ_ROOT" "$TRAIN_DATA" "$VAL_DATA" "$CACHE_ROOT"; do
  if [ ! -d "$required_dir" ]; then
    echo "Missing required directory: $required_dir" >&2
    exit 1
  fi
done

cd "$UNITRAJ_ROOT"

python unitraj/train.py \
  method=MTR \
  debug=False \
  'devices=[0]' \
  "train_data_path=[$TRAIN_DATA]" \
  "val_data_path=[$VAL_DATA]" \
  cache_path="$CACHE_ROOT" \
  use_cache=True \
  load_num_workers="$LOAD_NUM_WORKERS" \
  method.max_epochs="$MAX_EPOCHS" \
  exp_name="$EXP_NAME"
