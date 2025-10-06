#!/bin/bash
#SBATCH --out=classify.out
#SBATCH --err=classify.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1 ##### < change
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --partition=boost_usr_prod 
#SBATCH -A PHD_cozzani
#SBATCH 
#SBATCH --time=05:30:00
#SBATCH --mem=0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alberto.cozzani@studio.unibo.it

module purge
source /leonardo/home/userexternal/acozzani/my_venv/bin/activate


cd /leonardo_work/PHD_cozzani/seg_solar
export PYTHONPATH=$PWD:$PYTHONPATH

echo 'all modules loaded'

python tools/classify.py

