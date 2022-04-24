#!/usr/bin/env bash

experiment="resnet50_nodim_tin_final"
notes="
**Goal**: understand whether resnet50 >> resnet18 when the dimension is the same
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10080

kwargs="
experiment=$experiment
$base_kwargs_tin
representor=cntr
data_repr.kwargs.batch_size=512
architecture@encoder=resnet50
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
timeout=$time
update_trainer_repr.max_epochs=500
encoder.z_shape=2048
encoder.kwargs.arch_kwargs.is_channel_out_dim=True
+encoder.kwargs.arch_kwargs.bottleneck_channel=256
++decodability.kwargs.projector_kwargs.in_shape=2048
encoder.rm_out_chan_aug=True
seed=1
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "encoder.z_shape=2048 ++decodability.kwargs.projector_kwargs.in_shape=2048 architecture@encoder=resnet50,resnet18" "encoder.z_shape=512 ++decodability.kwargs.projector_kwargs.in_shape=512 architecture@encoder=resnet50,resnet18"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi_large $kwargs_dep $add_kwargs  -m  >> logs/"$experiment".log 2>&1 &


  done
fi

wait

# for representor
#python utils/aggregate.py \
#       experiment=$experiment  \
#       $col_val_subset \
#       agg_mode=[summarize_metrics]
