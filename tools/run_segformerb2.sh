#!/bin/bash
#SBATCH --out=Segformerb2.out
#SBATCH --err=Segformerb2.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=4 ##### < change
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod 
#SBATCH -A PHD_cozzani
#SBATCH 
#SBATCH --time=08:30:00
#SBATCH --mem=0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alberto.cozzani@studio.unibo.it


source /leonardo/home/userexternal/acozzani/my_venv/bin/activate
     

cd /leonardo_work/PHD_cozzani/seg_solar
export PYTHONPATH=$PWD:$PYTHONPATH

echo 'all modules loaded'

accelerate launch tools/train_segformerb2.py
