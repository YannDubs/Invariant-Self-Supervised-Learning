#!/usr/bin/env bash

experiment="swav_tin_final"
notes="
**Goal**: ensure that you can replicate swav on tinyimagenet.
"
# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh


time=10080

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
representor=slfdstl_swav
decodability.kwargs.temperature=0.07
decodability.kwargs.n_Mx=500
decodability.kwargs.epoch_queue_starts=5
timeout=$time
"

kwargs_multi="
seed=1,2,3
trainer.max_epochs=500
" # RUNNING STANFORD

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 10

  done
fi