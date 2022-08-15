#!/usr/bin/env bash

experiment="table_mlp_new"
notes="
**Goal**: run the main table for MLP.
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
representor=dstl_noema_mlp
downstream_task.all_tasks=[torchmlpw1e-4_datarepr,torchmlpw1e-5_datarepr,torchmlpw1e-6_datarepr,torchmlpw1e-3_datarepr002test,torchmlpw1e-5_datarepr002test,torchmlpw1e-4_datarepr002test,torchmlp_datarepr002test]
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "seed=2" #"$cell_baseline" #"$cell_reg seed=2,3,1" #"$cell_baseline"  #"$cell_linear" #
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

#python utils/aggregate.py \
#       experiment=$experiment  \
#       agg_mode=[summarize_metrics] \
#       $add_kwargs
