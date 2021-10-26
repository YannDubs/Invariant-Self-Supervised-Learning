#!/usr/bin/env bash

experiment=$prfx"generative_lr"
notes="
**Goal**: Hyperparameter tuning of learning rate.
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
representor=gen_no_norm
optimizer_issl.kwargs.lr=3e-4,1e-3,3e-3,1e-2,3e-2,1e-1
seed=1
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3

  done
fi
