#!/usr/bin/env bash

experiment="resnet50_cntr_tin"
notes="
**Goal**: tuning cntr resnet50 on tinyimagenet.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10080

kwargs="
experiment=$experiment
$base_kwargs_tin
representor=cntr
seed=1
data_repr.kwargs.batch_size=512
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
timeout=$time
"

kwargs_multi="
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "regularizer=etf representor.loss.beta=1e-1,1e-3" "regularizer=none,etf,huber,effdim representor.loss.beta=1e-2" "decodability.kwargs.predictor_kwargs.out_shape=128,512"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs  -m  >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi