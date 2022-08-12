#!/usr/bin/env bash

experiment="size_tin_final"
notes="
**Goal**: effect encoder size.
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
data_repr.kwargs.batch_size=192
representor=dstl
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
timeout=$time
"

kwargs_multi="
is_rescale_lr=True
encoder.kwargs.arch_kwargs.is_channel_out_dim=True
"

if [ "$is_plot_only" = false ] ; then
  # TODO run the same size without fixing dimension to show that dimension change is very important

  for kwargs_dep in "architecture@encoder=convnext_tiny encoder.z_shape=768"  #"architecture@encoder=convnext_tiny,convnext_small,convnext_base" "architecture@encoder=convnext_large data_repr.kwargs.batch_size=160"   # "encoder.z_shape=768 architecture@encoder=convnext_small,convnext_tiny" "encoder.z_shape=1024 architecture@encoder=convnext_base" "encoder.z_shape=1536 architecture@encoder=convnext_large data_repr.kwargs.batch_size=160" # "representor=slfdstl_dino encoder.z_shape=1024 architecture@encoder=convnext_base data_repr.kwargs.batch_size=160" #    #  # "representor=slfdstl_dino encoder.z_shape=1536 architecture@encoder=convnext_large" #  "encoder.z_shape=768 architecture@encoder=convnext_small,convnext_tiny" # "encoder.z_shape=768 architecture@encoder=convnext_small,convnext_tiny" #  # "representor=slfdstl_dino encoder.z_shape=1536 architecture@encoder=convnext_large data_repr.kwargs.batch_size=96"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs  -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

# TODO
# line plot. x = ISSL loss, y=acc

# line plot. x = paraemter, y = acc, hue: fixed vs varying dim