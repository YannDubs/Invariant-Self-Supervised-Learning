#!/usr/bin/env bash

experiment=$prfx"contrastive_lr_sched"
notes="
**Goal**: Hyperparameter tuning lr for contrastive ISSL.
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
timeout=$time

"


# every arguments that you are sweeping over
kwargs_multi="
representor=cntr,cntr_stdA
optimizer_issl.kwargs.lr=1e-4,3e-4,1e-3,3e-3,1e-2
scheduler@scheduler_issl=unifmultistep100,cosine,expdecay100,unifmultistep1000,plateau,plateau_quick
seed=1
"
#3e-4 or 1e-4 is good. slightly better 1e-4
# schedulers don't make big impact but unifmultistep100,cosine,plateau_quick seem robust



if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 3

  done
fi
