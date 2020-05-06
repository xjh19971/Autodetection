#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:k80:1
#SBATCH --time=08:00:00
#SBATCH --mem=100000
#SBATCH --job-name=xl1575
#SBATCH --mail-user=xl1575@nyu.edu
#SBATCH --output=slurm_%j.out


#command line argument
module load cudnn/9.0v7.3.0.29
module load cuda/9.0.176
. ~/.bashrc
module load anaconda3/5.3.1

conda activate cv
conda install -n cv nb_conda_kernels
# conda activate 

cd 
cd /scratch/xl1575/Autodetection/
python training/pretrain.py> pretrain_3.out
