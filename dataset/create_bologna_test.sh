#!/bin/bash
#SBATCH --out=create_test.out
#SBATCH --err=create_test.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1 ##### < change
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0
#SBATCH --partition=boost_usr_prod 
#SBATCH -A PHD_cozzani
#SBATCH --time=02:45:00
#SBATCH --mem=0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alberto.cozzani@studio.unibo.it


source /leonardo/home/userexternal/acozzani
cd /leonardo_work/PHD_cozzani/seg_solar/

python dataset/create_tiles.py
python dataset/compress_tiles.py
