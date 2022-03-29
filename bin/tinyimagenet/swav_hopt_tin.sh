#!/usr/bin/env bash

experiment="swav_hopt_tin"
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
decodability.kwargs.n_Mx=16384
decodability.kwargs.epoch_queue_starts=5
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
timeout=$time
"

kwargs_multi="
seed=1
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "decodability.kwargs.n_Mx=8192,16384,32768" "decodability.kwargs.epoch_queue_starts=0,5,15"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi