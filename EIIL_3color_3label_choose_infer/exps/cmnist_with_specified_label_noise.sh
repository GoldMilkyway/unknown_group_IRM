#!/bin/bash
# CMNIST Experiment.

# Hyperparameters
N_RESTARTS=10
HIDDEN_DIM=390
L2_REGULARIZER_WEIGHT=0.00110794568
LR=0.0004898536566546834
LABEL_NOISE=${1-0.05}
PENALTY_ANNEAL_ITERS=190
PENALTY_WEIGHT=191257.18613115903
STEPS=501
INFER=${2-3} ###
ROOT=${3-/scratch/gobi1/creager/opt_env/cmnist/EIIL_Research_results/EIIL_3color_3label_choose_infer}
RNUM=$(printf "%05d" $(($RANDOM$RANDOM$RANDOM % 100000)))
TAG=$(date +'%Y-%m-%d')--$LABEL_NOISE
ROOT=$ROOT/infer$INFER/label_noise_sweep/$TAG

# EIIL
python -u -m opt_env.irm_cmnist \
  --results_dir $ROOT/eiil \
  --n_restarts $N_RESTARTS \
  --hidden_dim $HIDDEN_DIM \
  --l2_regularizer_weight $L2_REGULARIZER_WEIGHT \
  --lr $LR \
  --label_noise $LABEL_NOISE \
  --penalty_anneal_iters $PENALTY_ANNEAL_ITERS \
  --penalty_weight $PENALTY_WEIGHT \
  --steps $STEPS \
  --eiil \
  --infer $INFER ###

# Grayscale baseline
python -u -m opt_env.irm_cmnist \
  --results_dir $ROOT/gray \
  --n_restarts $N_RESTARTS \
  --hidden_dim $HIDDEN_DIM \
  --l2_regularizer_weight $L2_REGULARIZER_WEIGHT \
  --lr $LR \
  --label_noise $LABEL_NOISE \
  --penalty_anneal_iters $PENALTY_ANNEAL_ITERS \
  --penalty_weight 0.0 \
  --steps $STEPS \
  --grayscale_model \
  --infer $INFER ###

# ERM
python -u -m opt_env.irm_cmnist \
  --results_dir $ROOT/erm \
  --n_restarts $N_RESTARTS \
  --hidden_dim $HIDDEN_DIM \
  --l2_regularizer_weight $L2_REGULARIZER_WEIGHT \
  --lr $LR \
  --label_noise $LABEL_NOISE \
  --penalty_anneal_iters 0 \
  --penalty_weight 0.0 \
  --steps $STEPS \
  --infer $INFER

# IRM
python -u -m opt_env.irm_cmnist \
  --results_dir $ROOT/irm \
  --n_restarts $N_RESTARTS \
  --hidden_dim $HIDDEN_DIM \
  --l2_regularizer_weight $L2_REGULARIZER_WEIGHT \
  --lr $LR \
  --label_noise $LABEL_NOISE \
  --penalty_anneal_iters $PENALTY_ANNEAL_ITERS \
  --penalty_weight $PENALTY_WEIGHT \
  --steps $STEPS \
  --infer $INFER


# Build latex tables
# accuracy
python -u -m opt_env.cmnist_results.acc_table \
  --erm_results_dir $ROOT/erm \
  --irm_results_dir $ROOT/irm \
  --eiil_results_dir $ROOT/eiil \
  --gray_results_dir $ROOT/gray \
  --results_dir $ROOT/acc_table
  