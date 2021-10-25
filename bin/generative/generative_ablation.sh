#!/usr/bin/env bash

experiment=$prfx"generative_ablation"
notes="
**Goal**: Ablation study to understand how to improve generative ISSL compared to standard for linear classification.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
trainer.max_epochs=50
checkpoint@checkpoint_repr=bestValLoss
architecture@encoder=resnet18
architecture@online_evaluator=linear
data@data_repr=mnist
data_pred.all_data=[data_repr_agg,data_repr_30,data_repr_100,data_repr_1000]
predictor=sk_logistic
encoder.z_shape=128
timeout=$time
$add_kwargs
"


# every arguments that you are sweeping over
kwargs_multi="
representor=std_gen,vae,gen,gen_no_norm,gen_no_V,gen_A_pred,gen_no_reg,gen_no_aug,gen_std_aug,gen_stoch,gen_Mx
seed=1
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3

  done
fi
