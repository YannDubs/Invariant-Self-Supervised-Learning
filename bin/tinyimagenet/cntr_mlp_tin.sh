#!/usr/bin/env bash

experiment="cntr_mlp_tin_final"
notes="
**Goal**:  contrastive on tinyimagenet.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10000

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
representor=cntr_mlp
architecture@online_evaluator=mlp
timeout=$time
"


# every arguments that you are sweeping over
kwargs_multi="
seed=1,2,3
downstream_task.all_tasks=[torchmlp_datarepr,torchmlpw1e-5_datarepr,torchmlpw1e-4_datarepr,torchlogisticw1e-4_datarepr]
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi
