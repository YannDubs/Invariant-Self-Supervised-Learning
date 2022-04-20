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
+encoder.kwargs.arch_kwargs.bottleneck_channel=512
encoder.rm_out_chan_aug=True
timeout=$time
"

MLP_bottleneck_postlinear="+decodability.kwargs.projector_kwargs.MLP_bottleneck_postlinear=512"
bttle_expand="encoder.rm_out_chan_aug=False"
bttle_expand="encoder.rm_out_chan_aug=False +decodability.kwargs.projector_kwargs.MLP_bottleneck_prelinear=512"
MLP_bottleneck_postlinear="decodability.kwargs.predictor_kwargs.bottleneck_size=null,128"
MLP_bottleneck_postlinear="decodability.kwargs.predictor_kwargs.is_train_bottleneck=False"
MLP_bottleneck_postlinear="decodability.kwargs.predictor_kwargs.is_batchnorm_bottleneck=True"
MLP_bottleneck_postlinear="decodability.kwargs.predictor_kwargs.batchnorm_kwargs.affine=True"



kwargs_multi="
seed=1
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "update_trainer_repr.max_epochs=500" "regularizer=huber representor.loss.beta=1e-5,1e-6"  "regularizer=rel_l1_clamp,effdim representor.loss.beta=1e-1,1e-2" # "" "+decodability.kwargs.projector_kwargs.MLP_bottleneck_postlinear=512" "encoder.rm_out_chan_aug=False" "encoder.rm_out_chan_aug=False +decodability.kwargs.projector_kwargs.MLP_bottleneck_prelinear=512" "+decodability.kwargs.predictor_kwargs.bottleneck_size=null,128" "decodability.kwargs.predictor_kwargs.is_train_bottleneck=False" "decodability.kwargs.predictor_kwargs.is_batchnorm_bottleneck=True" "decodability.kwargs.predictor_kwargs.batchnorm_kwargs.affine=True"
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
