#!/usr/bin/env bash

experiment="etf_tin"
notes="
**Goal**:  etf on tinyimagenet.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10000

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
representor=etf
seed=1
data_repr.kwargs.batch_size=512
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
timeout=$time
"

# every arguments that you are sweeping over
kwargs_multi="
"
# seed=1,2,3

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "encoder.is_etf_rep=True,False representor=etf" # "encoder.is_etf_rep=True representor=cntr" "encoder.is_etf_rep=True representor=dstl data_repr.kwargs.batch_size=256"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi
