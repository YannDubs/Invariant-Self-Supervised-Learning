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


rm_out_chan_aug="encoder.rm_out_chan_aug=True +encoder.kwargs.arch_kwargs.bottleneck_mode=mlp"
is_train_bottleneck="+decodability.kwargs.projector_kwargs.is_train_bottleneck=False +decodability.kwargs.projector_kwargs.kwargs_prelinear.is_train_bottleneck=False"
MLP_bottleneck_prelinear="+decodability.kwargs.projector_kwargs.MLP_bottleneck_prelinear=512"
MLP_bottleneck_prelinear="+decodability.kwargs.projector_kwargs.MLP_bottleneck_postlinear=512"
bottleneck_mode="+encoder.kwargs.arch_kwargs.bottleneck_channel=512 +encoder.kwargs.arch_kwargs.bottleneck_mode=cnn,mlp"
no_bottleneck="decodability.kwargs.projector_kwargs.bottleneck_size=null"
affine_false="+decodability.kwargs.projector_kwargs.kwargs_prelinear.bottleneck_size=512 +decodability.kwargs.projector_kwargs.kwargs_prelinear.batchnorm_kwargs.affine=False +decodability.kwargs.projector_kwargs.kwargs_prelinear.is_batchnorm_bottleneck=True"
smaller_bottleneck_pred="decodability.kwargs.predictor_kwargs.bottleneck_size=128"
no_chan="encoder.kwargs.arch_kwargs.is_channel_out_dim=False"
affine_pre="decodability.kwargs.is_batchnorm_pre=True +decodability.kwargs.batchnorm_kwargs.affine=False,True"
# "+encoder.kwargs.arch_kwargs.bottleneck_channel=512 +encoder.kwargs.arch_kwargs.is_bn_bttle_channel=True,False" # "+decodability.kwargs.projector_kwargs.is_batchnorm_bottleneck=False"  "+decodability.kwargs.predictor_kwargs.is_batchnorm_bottleneck=False"    "decodability.kwargs.projector_kwargs.bottleneck_size=null" #  ""  "+decodability.kwargs.projector_kwargs.kwargs_prelinear.bottleneck_size=512 +decodability.kwargs.projector_kwargs.kwargs_prelinear.is_batchnorm_bottleneck=True,False" "decodability.kwargs.predictor_kwargs.bottleneck_size=null"


kwargs_multi="
seed=1
"


if [ "$is_plot_only" = false ] ; then
  # TODO the second experiment now uses bottheleck size for projector => make sure it works
  for kwargs_dep in "+decodability.kwargs.projector_kwargs.MLP_bottleneck_postlinear=512" # "encoder.rm_out_chan_aug=True +encoder.kwargs.arch_kwargs.bottleneck_mode=mlp" "+decodability.kwargs.projector_kwargs.is_train_bottleneck=False +decodability.kwargs.projector_kwargs.kwargs_prelinear.is_train_bottleneck=False" "+decodability.kwargs.projector_kwargs.MLP_bottleneck_prelinear=512" ""  "+encoder.kwargs.arch_kwargs.bottleneck_channel=512 +encoder.kwargs.arch_kwargs.bottleneck_mode=cnn,mlp" "decodability.kwargs.projector_kwargs.bottleneck_size=null" "+decodability.kwargs.projector_kwargs.kwargs_prelinear.bottleneck_size=512 +decodability.kwargs.projector_kwargs.kwargs_prelinear.batchnorm_kwargs.affine=False +decodability.kwargs.projector_kwargs.kwargs_prelinear.is_batchnorm_bottleneck=True" "decodability.kwargs.predictor_kwargs.bottleneck_size=128" "encoder.kwargs.arch_kwargs.is_channel_out_dim=False" "decodability.kwargs.is_batchnorm_pre=True +decodability.kwargs.batchnorm_kwargs.affine=False,True"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs  -m >> logs/"$experiment".log 2>&1 &

    sleep 5

  done
fi

wait

# for representor
#python utils/aggregate.py \
#       experiment=$experiment  \
#       $col_val_subset \
#       agg_mode=[summarize_metrics]
