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
timeout=$time
"

kwargs_multi="
seed=1
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "+decodability.kwargs.projector_kwargs.is_batchnorm_bottleneck=False"  "+decodability.kwargs.predictor_kwargs.is_batchnorm_bottleneck=False"    "decodability.kwargs.projector_kwargs.bottleneck_size=null" #  ""  "+decodability.kwargs.projector_kwargs.kwargs_prelinear.bottleneck_size=512 +decodability.kwargs.projector_kwargs.kwargs_prelinear.is_batchnorm_bottleneck=True,False" "decodability.kwargs.predictor_kwargs.bottleneck_size=null"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs  -m # >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

wait

# for representor
python utils/aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       agg_mode=[summarize_metrics]
