#!/usr/bin/env bash

experiment="dstl_tin_final"
notes="
**Goal**: distillation final on timy imagenet.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh



time=10080

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
representor=dstl
data_repr.kwargs.batch_size=256
timeout=$time
"

kwargs_multi="
seed=1,2,3
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr,torchlogisticw1e-5_datarepr001test,torchlogisticw1e-4_datarepr001test,torchlogisticw1e-3_datarepr001test,sklogistic_datarepr,sklogisticreg01_datarepr,sklogisticreg001_datarepr]
"

kwargs_multi="
seed=1
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
data_repr.kwargs.batch_size=256,512
"

kwargs_multi="
seed=1
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
update_trainer_repr.max_epochs=300,500
"



if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in    ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs  -m  >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

wait

# for representor
python utils/aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       agg_mode=[summarize_metrics]
