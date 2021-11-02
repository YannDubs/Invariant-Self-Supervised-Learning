#!/usr/bin/env bash

experiment=$prfx"supervised"
notes="
**Goal**: supervised baseline.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
trainer.max_epochs=50
data@data_repr=mnist
data_pred.all_data=[data_repr_agg,data_repr_30,data_repr_100,data_repr_100_test,data_repr_1000]
representor=none
predictor=pytorch
timeout=$time
$add_kwargs
"


# every arguments that you are sweeping over
kwargs_multi="
architecture@predictor=resnet18
seed=1
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3

  done
fi
