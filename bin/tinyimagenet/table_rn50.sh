#!/usr/bin/env bash

experiment="table_rn50"
notes="
**Goal**: run resnet50 dissl.
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
representor=dissl
downstream_task.all_tasks=[torchlogisticw1e-4_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-6_datarepr]
data_repr.kwargs.batch_size=256
"

cell_rn_50="
encoder.z_dim=2048
architecture@encoder=resnet50
"

cell_dim="
$cell_rn_50
encoder.z_dim=512
encoder.kwargs.arch_kwargs.is_channel_out_dim=True
+encoder.kwargs.arch_kwargs.bottleneck_channel=512
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "$cell_rn_50"  #"$cell_dim"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi