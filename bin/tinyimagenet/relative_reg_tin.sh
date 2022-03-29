#!/usr/bin/env bash

experiment="relative_reg_tin"
notes="
**Goal**: make regularizer work using a relative one to avoid all to 0.
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
timeout=$time
"

# every arguments that you are sweeping over
kwargs_multi="
representor=dstl
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
regularizer=rel_l1
representor.loss.beta=0.01,0.1,1,10
update_trainer_repr.max_epochs=100
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

# TODO add plotting with `*` to take the baest over torchlogistic*_datarepr