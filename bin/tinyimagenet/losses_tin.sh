#!/usr/bin/env bash

experiment="losses_tin_final"
notes="
**Goal**: understanding effect of using different losses that do not imply invariance.
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
data_repr.kwargs.batch_size=512
timeout=$time
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
"

# every arguments that you are sweeping over
kwargs_multi="
seed=1
decodability.kwargs.loss=mse,ce,margin
decodability.kwargs.is_self_contrastive=True,False
"

kwargs_multi="
seed=1
decodability.kwargs.loss=mse
decodability.kwargs.is_self_contrastive=True,False
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi
