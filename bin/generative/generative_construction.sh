#!/usr/bin/env bash

experiment=$prfx"generative_construction"
notes="
**Goal**: Ablation study to understand how to improve generative ISSL compared to standard for linear classification.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
trainer.max_epochs=50
checkpoint@checkpoint_repr=bestTrainLoss
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
representor=vae,std_gen_V,std_gen_supA,std_gen_stoch,std_gen_stdA_pred,std_gen_stdA,std_gen_reg,std_gen_permMx,std_gen_permA,std_gen_norm,std_gen_Mx,std_gen_mlp,std_gen,gen
seed=1
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3

  done
fi
