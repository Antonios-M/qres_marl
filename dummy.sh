#!/bin/sh
#
#SBATCH --job-name=setup_conda_env
#SBATCH --partition=compute
#SBATCH --account=Education-ABE-MSc-A
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1GB

module load miniconda3
conda activate ./conda_env
srun python dummy.py
conda deactivate
