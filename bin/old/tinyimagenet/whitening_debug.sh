#!/usr/bin/env bash

experiment="whitening_debug"
notes="
**Goal**: run the main table for contrastive.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10000

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
seed=1
timeout=$time
downstream_task.all_tasks=[torchlogisticw1e-4_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw3e-6_datarepr,torchlogisticw3e-5_datarepr]
seed=1
++decodability.kwargs.projector_kwargs.n_hid_layers=1
++decodability.kwargs.projector_kwargs.hid_dim=1024
"

kwargs_multi="
representor=cntr,cntr_blured,cntr_whitening,cntr_normalized
"

kwargs_multi="
representor=cntr_normalized
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in   ""
  do
    # 3681265 - 3681273
    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

#python utils/aggregate.py \
#       experiment=$experiment  \
#       agg_mode=[summarize_metrics] \
#       $add_kwargs
