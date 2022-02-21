#!/usr/bin/env bash

experiment="cntr_hopt_tin_final"
notes="
**Goal**: hyperparameter tuning for contrastive on tinyimagenet.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10000

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
representor=cntr
decodability.kwargs.temperature=0.07
trainer.max_epochs=500
timeout=$time
"


# every arguments that you are sweeping over
kwargs_multi="
seed=1
"
#seed=1,2,3
# /juice/scr/yanndubs/Invariant-Self-Supervised-Learning/outputs/2022-02-21_15-26-55

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 10

  done
fi
