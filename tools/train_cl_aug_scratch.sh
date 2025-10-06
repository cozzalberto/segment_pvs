#!/bin/bash
#SBATCH --out=train_classifier_scratch.out
#SBATCH --err=train_classifier_scratch.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=4 ##### < change
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod 
#SBATCH -A PHD_cozzani
#SBATCH 
#SBATCH --time=06:15:00
#SBATCH --mem=0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alberto.cozzani@studio.unibo.it

module purge
source /leonardo/home/userexternal/acozzani/my_venv/bin/activate
       

cd /leonardo_work/PHD_cozzani/seg_solar
export PYTHONPATH=$PWD:$PYTHONPATH
echo 'all modules loaded'

torchrun --nproc_per_node=4 tools/train_cl_aug_scratch.py

