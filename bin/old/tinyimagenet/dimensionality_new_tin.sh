#!/usr/bin/env bash

experiment="dimensionality_new_tin"
notes="
**Goal**: effect of using larger dimension Z.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10000

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
representor=dstl
data_repr.kwargs.batch_size=256
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
encoder.kwargs.arch_kwargs.is_channel_out_dim=True
encoder.z_shape=8192
update_trainer_repr.max_epochs=500
+encoder.kwargs.arch_kwargs.bottleneck_channel=512
seed=1
timeout=$time
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in   "+decodability.kwargs.projector_kwargs.in_shape=512 encoder.rm_out_chan_aug=True" "encoder.rm_out_chan_aug=False"
  do
    for kwargs_multi in   "data_repr.kwargs.batch_size=256 representor=dstl" "data_repr.kwargs.batch_size=512 representor=cntr"
    do

      python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

      sleep 10
    done

  done
fi
# TODO
# line plot. x = dim , y = acc, hue = repr
# color code 512 to say what is typical



