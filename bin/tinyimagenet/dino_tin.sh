#!/usr/bin/env bash

experiment="dino_tin_final"
notes="
**Goal**: ensure that you can replicate dino on tinyimagenet.
"
# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh


time=10000

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
representor=slfdstl_dino
data_repr.kwargs.batch_size=256
timeout=$time
"

kwargs_multi="
seed=1,2,3
trainer.max_epochs=500
"
# /juice/scr/yanndubs/Invariant-Self-Supervised-Learning/outputs/2022-02-22_16-06-07
# need to continue seed 1

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 10

  done
fi