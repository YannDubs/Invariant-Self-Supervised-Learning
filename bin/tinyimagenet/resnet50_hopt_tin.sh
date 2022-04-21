#!/usr/bin/env bash

experiment="resnet50_hopt_tin"
notes="
**Goal**: understand how to increase diemnsionality with resnet50.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10080

kwargs="
experiment=$experiment
$base_kwargs_tin
representor=dstl
data_repr.kwargs.batch_size=256
architecture@encoder=resnet50
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
timeout=$time
update_trainer_repr.max_epochs=200
encoder.z_shape=2048
decodability.kwargs.predictor_kwargs.bottleneck_size=256
seed=1
"

kwargs_multi_large="
encoder.z_shape=8192
encoder.kwargs.arch_kwargs.is_channel_out_dim=True
+encoder.kwargs.arch_kwargs.bottleneck_channel=256
+decodability.kwargs.projector_kwargs.in_shape=2048
encoder.rm_out_chan_aug=True
"



if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "" "decodability.kwargs.predictor_kwargs.bottleneck_size=512" "decodability.kwargs.predictor_kwargs.is_train_bottleneck=False" "regularizer=effdim,rel_l1_clamp representor.loss.beta=1e-1"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs kwargs_multi_large $kwargs_dep $add_kwargs  -m  >> logs/"$experiment".log 2>&1 &

    sleep 10

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_dep $add_kwargs  -m  >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

wait

# for representor
#python utils/aggregate.py \
#       experiment=$experiment  \
#       $col_val_subset \
#       agg_mode=[summarize_metrics]
