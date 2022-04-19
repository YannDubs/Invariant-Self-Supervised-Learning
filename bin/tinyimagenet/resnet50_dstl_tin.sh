#!/usr/bin/env bash

experiment="resnet50_dstl_tin"
notes="
**Goal**: tuning resnet50  on tiny imagenet to chose parameters for imagenet.
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
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
+encoder.kwargs.arch_kwargs.bottleneck_channel=512
+decodability.kwargs.projector_kwargs.kwargs_prelinear.bottleneck_size=512
timeout=$time
update_trainer_repr.max_epochs=200
encoder.z_shape=2048
"

kwargs_multi="
seed=1
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "" "encoder.z_shape=8192 encoder.kwargs.arch_kwargs.is_channel_out_dim=True" #  "encoder.z_shape=4096,8192,16384"#  "representor.loss.beta=1e-5" #
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs  -m  >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

wait

# for representor
python utils/aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       agg_mode=[summarize_metrics]
