#!/usr/bin/env bash


experiment="simclr_tin_final"
notes="
**Goal**: ensure that you can replicate the whitening paper for tinyimagenet with simclr.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10000

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
representor=cntr_simclr
decodability.kwargs.temperature=0.07
timeout=$time
"

kwargs_multi="
seed=1,2,3
trainer.max_epochs=500
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 10

  done
fi
