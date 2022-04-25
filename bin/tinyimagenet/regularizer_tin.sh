#!/usr/bin/env bash

experiment="regularizer_tin_final"
notes="
**Goal**: effect of regularizer on computational efficiency.
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
data_repr.kwargs.batch_size=256
representor=dstl
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
timeout=$time
"

# every arguments that you are sweeping over
kwargs_multi="
update_trainer_repr.max_epochs=200,500
"

kwargs_multi="
update_trainer_repr.max_epochs=200
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "regularizer=etf_approx,etf_cc,etf_both representor.loss.beta=1e-1"  # "regularizer=etf representor.loss.beta=1e-3,1e-2,1e-1,1,10" #"regularizer=huber representor.loss.beta=1e-5" "regularizer=rel_l1_clamp representor.loss.beta=1e-1" ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

# TODO
# line plot. x = number of downstream samples, y=acc, hue=epoch, style=beta / is_reg