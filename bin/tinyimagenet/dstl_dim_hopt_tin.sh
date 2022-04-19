#!/usr/bin/env bash

experiment="dstl_dim_hopt_tin"
notes="
**Goal**: understand how to make dstl work with large dim.
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
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
encoder.kwargs.arch_kwargs.is_channel_out_dim=True
encoder.z_shape=512
update_trainer_repr.max_epochs=200
timeout=$time
"


"encoder.rm_out_chan_aug=True encoder.kwargs.arch_kwargs.is_channel_out_dim=True +encoder.kwargs.arch_kwargs.bottleneck_channel=128 +encoder.kwargs.arch_kwargs.bottleneck_mode=cnn encoder.z_shape=2048 +decodability.kwargs.projector_kwargs.in_shape=512 decodability.kwargs.predictor_kwargs.bottleneck_size=256"


kwargs_multi="
seed=1
"


if [ "$is_plot_only" = false ] ; then
  # TODO the second experiment now uses bottheleck size for projector => make sure it works
  for kwargs_dep in  "+decodability.kwargs.projector_kwargs.MLP_bttle_prelinear=512" # "+encoder.kwargs.arch_kwargs.bottleneck_channel=512 +encoder.kwargs.arch_kwargs.bottleneck_mode=cnn,mlp" # "+encoder.kwargs.arch_kwargs.bottleneck_channel=512 +encoder.kwargs.arch_kwargs.bottleneck_mode=cnn,mlp" # "+encoder.kwargs.arch_kwargs.bottleneck_channel=512 +encoder.kwargs.arch_kwargs.is_bn_bttle_channel=True,False" # ""  "decodability.kwargs.projector_kwargs.bottleneck_size=null" "+decodability.kwargs.projector_kwargs.kwargs_prelinear.bottleneck_size=512 +decodability.kwargs.projector_kwargs.kwargs_prelinear.batchnorm_kwargs.affine=False +decodability.kwargs.projector_kwargs.kwargs_prelinear.is_batchnorm_bottleneck=True"  "decodability.kwargs.predictor_kwargs.bottleneck_size=128" "encoder.kwargs.arch_kwargs.is_channel_out_dim=False" "decodability.kwargs.is_batchnorm_pre=True +decodability.kwargs.batchnorm_kwargs.affine=False,True" # "+decodability.kwargs.projector_kwargs.is_batchnorm_bottleneck=False"  "+decodability.kwargs.predictor_kwargs.is_batchnorm_bottleneck=False"    "decodability.kwargs.projector_kwargs.bottleneck_size=null" #  ""  "+decodability.kwargs.projector_kwargs.kwargs_prelinear.bottleneck_size=512 +decodability.kwargs.projector_kwargs.kwargs_prelinear.is_batchnorm_bottleneck=True,False" "decodability.kwargs.predictor_kwargs.bottleneck_size=null"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs  -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

wait

# for representor
python utils/aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       agg_mode=[summarize_metrics]
