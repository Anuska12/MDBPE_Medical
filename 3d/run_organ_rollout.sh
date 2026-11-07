#!/bin/bash
#SBATCH --job-name=organ_rollout
#SBATCH --account=your_account
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=organ_rollout_%j.out
# usage: sbatch run_organ_rollout.sh <seed>
module load python
source ~/mdbpe_env/bin/activate
cd ~/your/path/3d
python3 -u organmnist3d_attention_rollout.py "$@"
