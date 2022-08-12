#!/usr/bin/env bash

experiment="cntr_temp_tin"
notes="
**Goal**: tuning temperature to get ETF quicker.
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
seed=1
data_repr.kwargs.batch_size=512
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
update_trainer_repr.max_epochs=200
timeout=$time
"

# every arguments that you are sweeping over
kwargs_multi="
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "decodability.kwargs.is_cosine_pos_temperature=True" "decodability.kwargs.is_cosine_neg_temperature=True" "decodability.kwargs.is_train_temperature=True" "" "decodability.kwargs.temperature=1.0"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi
