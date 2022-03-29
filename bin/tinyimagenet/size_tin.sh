#!/usr/bin/env bash

experiment="size_tin"
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
is_rescale_lr=True
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
timeout=$time
"

kwargs_multi="
representor=dstl
"

if [ "$is_plot_only" = false ] ; then
  # rescale lrs for more fair comparison when change batch size
  # TODO run dino
  for kwargs_dep in "encoder.z_shape=768 architecture@encoder=convnext_small,convnext_tiny" "encoder.z_shape=1024 architecture@encoder=convnext_base" "encoder.z_shape=1536 architecture@encoder=convnext_large data_repr.kwargs.batch_size=160" # "representor=slfdstl_dino encoder.z_shape=1024 architecture@encoder=convnext_base data_repr.kwargs.batch_size=160" #    #  # "representor=slfdstl_dino encoder.z_shape=1536 architecture@encoder=convnext_large" #  "encoder.z_shape=768 architecture@encoder=convnext_small,convnext_tiny" # "encoder.z_shape=768 architecture@encoder=convnext_small,convnext_tiny" #  # "representor=slfdstl_dino encoder.z_shape=1536 architecture@encoder=convnext_large data_repr.kwargs.batch_size=96"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs  -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi
