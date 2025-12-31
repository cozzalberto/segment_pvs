#!/bin/bash
#SBATCH --out=train_classifier.out
#SBATCH --err=ablation_classifier_cp.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=4 ##### < change
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH -A PHD_cozzani
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --mem=0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alberto.cozzani@studio.unibo.it

module purge
source /leonardo/home/userexternal/acozzani/my_venv/bin/activate
export PYTHONPATH="/leonardo_work/PHD_cozzani/seg_solar:$PYTHONPATH"

PROJECT_ROOT="/leonardo_work/PHD_cozzani/seg_solar/"
cd "${PROJECT_ROOT}"
dataset=danish_with_bbr_and_google
exp_name=efficientnet_v2s
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=configs/${dataset}/${dataset}_${exp_name}.yaml
export PYTHONPATH=${PYTHONPATH}:${PROJECT_ROOT}
mkdir -p ${model_dir} ${result_dir}
cp tools/ablation_study_cp.sh tools/ablation_study_cp.py tools/test.sh tools/test.py ${config} ${exp_dir}


torchrun --nproc_per_node=4 $exp_dir/ablation_study_cp.py  --config ${config}  ${@} 
