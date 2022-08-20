#!/usr/bin/env bash

experiment="table2_mlp"
notes="
**Goal**: evaluates MLP probing (table 3).
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10000

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
timeout=$time
representor=dissl_mlp
downstream_task.all_tasks=[torchmlpw1e-4_datarepr,torchmlpw1e-5_datarepr,torchmlpw1e-6_datarepr,torchmlpw1e-3_datarepr002test,torchmlpw1e-5_datarepr002test,torchmlpw1e-4_datarepr002test,torchmlp_datarepr002test]
seed=1
"
# use seed=1,2,3 for the three seeds

# to get the other models we simply evaluated the oens that were pretrained in `table_dstl_clean.sh`

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "decodability.kwargs.lambda_maximality=2.5,2.7" # ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

#python utils/aggregate.py \
#       experiment=$experiment  \
#       agg_mode=[summarize_metrics] \
#       $add_kwargs
