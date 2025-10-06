#!/bin/bash
#SBATCH --out=train_classifier.out
#SBATCH --err=train_classifier.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=4 ##### < change
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod 
#SBATCH -A PHD_cozzani
#SBATCH 
#SBATCH --time=01:35:00
#SBATCH --mem=0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alberto.cozzani@studio.unibo.it

module purge
source /leonardo/home/userexternal/acozzani/my_venv/bin/activate
       

cd /leonardo_work/PHD_cozzani/seg_solar
export PYTHONPATH=$PWD:$PYTHONPATH
echo 'all modules loaded'

torchrun --nproc_per_node=4 tools/train_classifier.py

