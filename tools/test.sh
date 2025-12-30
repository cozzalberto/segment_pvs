#!/bin/bash
#SBATCH --out=train_classifier.out
#SBATCH --err=train_classifier.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1 ##### < change
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod 
#SBATCH --qos=boost_qos_dbg 
#SBATCH -A PHD_cozzani
#SBATCH --time=00:04:00
#SBATCH --mem=0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alberto.cozzani@studio.unibo.it

module purge
source /leonardo/home/userexternal/acozzani/my_venv/bin/activate

export PYTHONPATH="/leonardo_work/PHD_cozzani/seg_solar:$PYTHONPATH"
cd /leonardo_work/PHD_cozzani/seg_solar/
MODEL_NAME=$1
echo 'all modules loaded'

python tools/test.py --model_name "$MODEL_NAME"


