#!/bin/bash
# Usage: ./setup_conda_env.sh

#SBATCH --job-name=setup_conda_env
#SBATCH --partition=compute
#SBATCH --account=Education-ABE-MSc-A
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1GB

module load miniconda3
ENV_DIR= /conda_env
conda env create -p "$ENV_DIR" -f conda_env.yml
