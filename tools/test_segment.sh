#!/bin/bash
#SBATCH --out=test_segmentsuherlev.out
#SBATCH --err=test_segment.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1 ##### < change
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod 
#SBATCH -A PHD_cozzani
#SBATCH 
#SBATCH --time=00:50:00
#SBATCH --mem=0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alberto.cozzani@studio.unibo.it

module purge
source /leonardo/home/userexternal/acozzani/my_venv/bin/activate
       

cd /leonardo_work/PHD_cozzani/seg_solar
export PYTHONPATH=$PWD:$PYTHONPATH
echo 'all modules loaded'

python tools/test_segment.py

