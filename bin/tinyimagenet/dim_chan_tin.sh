#!/usr/bin/env bash

experiment="dim_tin"
notes="
**Goal**:  effect of using larger dimension Z.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10000

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
seed=2
timeout=$time
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_multi in "encoder.kwargs.arch_kwargs.is_channel_out_dim=True encoder.z_shape=2048" # "encoder.kwargs.arch_kwargs.is_channel_out_dim=True encoder.z_shape=8192 +decodability.kwargs.projector_kwargs.kwargs_prelinear.bottleneck_size=512"
  do
    for kwargs_dep in "representor=dstl data_repr.kwargs.batch_size=256" # "representor=cntr data_repr.kwargs.batch_size=512"   #  #  #  #  #
    do

      python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

      sleep 10

    done
  done
fi
