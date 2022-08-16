#!/usr/bin/env bash

experiment="dissl_act"
notes="
**Goal**: impact of activation on dissl
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10000

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
seed=2
timeout=$time
representor=dstl
downstream_task.all_tasks=[torchlogisticw1e-4_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-6_datarepr,torchlogisticw1e-5b2048e300_datarepr]
++decodability.kwargs.projector_kwargs.n_hid_layers=1
++decodability.kwargs.projector_kwargs.hid_dim=1024
"

kwargs_multi="
+decodability.kwargs.projector_kwargs.activation=GELU,ReLU
"



if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi
#
#python utils/aggregate.py \
#       experiment=$experiment  \
#       agg_mode=[summarize_metrics] \
#       $add_kwargs