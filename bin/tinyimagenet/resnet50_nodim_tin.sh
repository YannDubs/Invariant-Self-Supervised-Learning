#!/usr/bin/env bash

experiment="resnet50_nodim_tin"
notes="
**Goal**: understnading whether resnet50 gets much gains besides dimensionality.
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
architecture@encoder=resnet50
seed=1
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr,torchlogisticw1e-3_datarepr]
encoder.z_shape=2048
timeout=$time
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_multi in  "" #"encoder.kwargs.arch_kwargs.is_channel_out_dim=True encoder.z_shape=512" #"" #
  do
    for kwargs_dep in   "architecture@encoder=resnet18 encoder.kwargs.arch_kwargs.is_channel_out_dim=True encoder.z_shape=2048" #"" # #"representor=cntr data_repr.kwargs.batch_size=512"
    do

      python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs  -m  >> logs/"$experiment".log 2>&1 &

      sleep 10
    done
  done
fi

# TODO table showing resnet50 / resnet18 AND 512 / 2048 dim AND cntr / dstl
# => show that all that matters is actually dimensionality