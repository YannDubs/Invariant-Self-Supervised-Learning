#!/usr/bin/env bash

experiment="table_dstl_new2"
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
representor=dstl_noema
downstream_task.all_tasks=[torchlogisticw1e-4_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-6_datarepr,torchlogisticw1e-5b2048e300_datarepr]
++decodability.kwargs.projector_kwargs.n_hid_layers=1
++decodability.kwargs.projector_kwargs.hid_dim=1024
data_repr.kwargs.batch_size=256
"

cell_baseline="
representor=slfdstl_dino
downstream_task.all_tasks=[torchlogisticw1e-4_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-6_datarepr,torchlogisticw1e-5b2048e300_datarepr,torchmlpw1e-4_datarepr,torchmlpw1e-5_datarepr,torchmlpw1e-6_datarepr,torchmlpw1e-5b2048e300_datarepr,torchmlpw1e-3_datarepr002test,torchmlpw1e-5_datarepr002test,torchmlpw1e-4_datarepr002test,torchmlp_datarepr002test]
"

cell_ours="
downstream_task.all_tasks=[torchlogisticw1e-4_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-6datarepr,torchlogisticw1e-5b2048e300_datarepr,torchmlpw1e-4_datarepr,torchmlpw1e-5_datarepr,torchmlpw1e-6_datarepr,torchmlpw1e-5b2048e300_datarepr,torchmlpw1e-3_datarepr002test,torchmlpw1e-5_datarepr002test,torchmlpw1e-4_datarepr002test,torchmlp_datarepr002test,torchlogisticw1e-4_datarepr002test,torchlogisticw1e-5_datarepr002test,torchlogisticw1e-3_datarepr002test,torchlogistic_datarepr002test]
"

cell_noJL="
decodability.kwargs.predictor_kwargs.is_JL_init=False
"


cell_large_proj="
++decodability.kwargs.projector_kwargs.n_hid_layers=2
++decodability.kwargs.projector_kwargs.hid_dim=2048
"

cell_dim="
encoder.z_shape=2048
encoder.kwargs.arch_kwargs.is_channel_out_dim=True
+encoder.kwargs.arch_kwargs.bottleneck_channel=512
"

cell_asymm_small="
$cell_dim
encoder.aux_enc_base=resnet18
"

cell_asymm_large="
$cell_dim
encoder.aux_enc_base=resnet50
++encoder.kwargs.arch_kwargs.is_resize_only_if_necessary=True
"

#cell_aug="
#$cell_asymm_large
#data_repr.kwargs.dataset_kwargs.simclr_aug_strength=2.0
#representor=dstl_noema_blured
#"
#
#cell_epoch="
#$cell_aug
#update_trainer_repr.max_epochs=1000
#"

cell_aug="
$cell_dim
data_repr.kwargs.dataset_kwargs.simclr_aug_strength=2.0
representor=dstl_noema_blured
"

cell_epoch="
$cell_aug
update_trainer_repr.max_epochs=1000
"

cell_epoch_coarse="
$cell_aug
update_trainer_repr.max_epochs=1000
decodability.kwargs.out_dim=8192
"

cell_epoch_noaug="
$cell_dim
update_trainer_repr.max_epochs=1000
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "$cell_noJL" #"$cell_large_proj" # "$cell_epoch_coarse" #"$cell_ours" #  #"$cell_ours"   # "$cell_aug" "$cell_epoch" "$cell_ours" "$cell_dim"   #  "$cell_ours" "$cell_dim" "$cell_asymm_small" "$cell_asymm_large" "$cell_aug" "$cell_epoch" # cell_baseline
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi
#
#python utils/aggregate.py \
#       experiment=$experiment  \
#       agg_mode=[summarize_metrics] \
#       $add_kwargs
