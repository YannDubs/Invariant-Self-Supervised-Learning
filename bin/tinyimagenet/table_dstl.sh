#!/usr/bin/env bash

experiment="table_dstl"
notes="
**Goal**: run the main table for distillation.
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
representor=dstl
data_repr.kwargs.batch_size=256
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
update_trainer_repr.max_epochs=500
seed=1
++decodability.kwargs.projector_kwargs.n_hid_layers=1
++decodability.kwargs.projector_kwargs.hid_dim=1024
"

cell_baseline="
representor=slfdstl_dino
"

cell_head="
++decodability.kwargs.projector_kwargs.n_hid_layers=2
++decodability.kwargs.projector_kwargs.hid_dim=2048
"

cell_reg_hopt="
$cell_head
regularizer=effdim,etf
representor.loss.beta=1e-2
"


cell_reg="
$cell_head
regularizer=etf
representor.loss.beta=1e-2
"

cell_dim_hopt1="
$cell_reg
encoder.z_shape=2048
encoder.rm_out_chan_aug=False
encoder.kwargs.arch_kwargs.is_channel_out_dim=True
+encoder.kwargs.arch_kwargs.bottleneck_channel=512
"

cell_dim_hopt2="
$cell_reg
encoder.z_shape=2048
encoder.rm_out_chan_aug=True
encoder.kwargs.arch_kwargs.is_channel_out_dim=True
+encoder.kwargs.arch_kwargs.bottleneck_channel=512
+decodability.kwargs.projector_kwargs.in_shape=512
"

cell_dim="$cell_dim_hopt2"

cell_aug="
$cell_dim
data_repr.kwargs.dataset_kwargs.simclr_aug_strength=2.0
"

cell_epoch="
$cell_aug
update_trainer_repr.max_epochs=1000
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "" "$cell_baseline"    "$cell_head" "$cell_reg_hopt" "$cell_dim_hopt1" "$cell_dim_hopt2" "$cell_aug" "$cell_epoch"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

#python utils/aggregate.py \
#       experiment=$experiment  \
#       agg_mode=[summarize_metrics] \
#       $add_kwargs
