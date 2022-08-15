#!/usr/bin/env bash

experiment="table_cntr_final"
notes="
**Goal**: run the main table for contrastive.
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
representor=cntr
downstream_task.all_tasks=[torchlogisticw1e-4_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-6_datarepr,torchlogisticw1e-5b2048e300_datarepr]
seed=1
++decodability.kwargs.projector_kwargs.n_hid_layers=1
++decodability.kwargs.projector_kwargs.hid_dim=1024
"

cell_baseline="
representor=cntr_simclr
"

cell_ours="
representor=cntr
"

cell_ours_noselfcntr="
representor=cntr
decodability.kwargs.is_self_contrastive=False
"

cell_dim="
encoder.z_shape=2048
encoder.kwargs.arch_kwargs.is_channel_out_dim=True
+encoder.kwargs.arch_kwargs.bottleneck_channel=512
"

cell_aug="
$cell_dim
data_repr.kwargs.dataset_kwargs.simclr_aug_strength=2.0
representor=cntr_blured
"

cell_aug_noblured="
$cell_dim
data_repr.kwargs.dataset_kwargs.simclr_aug_strength=2.0
"


cell_aug_onlyblured="
$cell_dim
representor=cntr_blured
"

cell_epoch="
$cell_aug
update_trainer_repr.max_epochs=1000
"

cell_epoch_noaug="
$cell_dim
update_trainer_repr.max_epochs=1000
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "$cell_epoch" # "$cell_aug_noblured" "$cell_aug_onlyblured"  #"$cell_baseline" "$cell_ours"  "$cell_dim"  "$cell_aug" "$cell_epoch"
  do
    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

#python utils/aggregate.py \
#       experiment=$experiment  \
#       agg_mode=[summarize_metrics] \
#       $add_kwargs
