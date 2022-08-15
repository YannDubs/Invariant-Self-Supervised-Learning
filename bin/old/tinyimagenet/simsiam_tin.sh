#!/usr/bin/env bash

experiment="simsiam_tin_final"
notes="
**Goal**: ensure that you can replicate simsiam on tinyimagenet.
"


# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10000

kwargs="
experiment=$experiment
$base_kwargs_tin
representor=slfdstl_simsiam
optimizer_issl.kwargs.weight_decay=1e-4
timeout=$time
"

kwargs_multi="
seed=1,2,3
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 10

  done
fi