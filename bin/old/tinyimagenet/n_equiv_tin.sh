#!/usr/bin/env bash

experiment="n_equiv_tin"
notes="
**Goal**: understanding effect of number of equivalence classes with DISSL.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh



time=10080

kwargs="
experiment=$experiment
$base_kwargs_tin
representor=dstl
data_repr.kwargs.batch_size=256
seed=1
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr,torchlogisticw1e-5_datarepr001test,torchlogisticw1e-4_datarepr001test,torchlogisticw1e-3_datarepr001test]
timeout=$time
"

kwargs_multi="
decodability.kwargs.out_dim=512,2048,8192,32768
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs  -m  >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

# TODO
# line plot. y = acc, x = n equiv
# => show that DISSL has a way of controlling number fo equiv class (coarsening) but then depends on inductive bias.