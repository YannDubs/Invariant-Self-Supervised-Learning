#!/usr/bin/env bash

experiment="resnet50_dstl_tin"
notes="
**Goal**: tuning resnet50 on tinyimagenet.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10080

kwargs="
experiment=$experiment
$base_kwargs_tin
representor=dstl
seed=1
data_repr.kwargs.batch_size=256
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
timeout=$time
"

kwargs_multi="
decodability.kwargs.predictor_kwargs.bottleneck_size=512
decodability.kwargs.out_dim=16384
representor.loss.beta=1e-2
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "decodability.kwargs.freeze_Mx_epochs=10" "decodability.kwargs.freeze_Mx_epochs=1 decodability.kwargs.is_freeze_only_bottleneck=False"  #"regularizer=etf representor.loss.beta=1e-1,1e-3" "regularizer=none,etf,huber,effdim" "decodability.kwargs.predictor_kwargs.is_train_bottleneck=False,True" "decodability.kwargs.ema_weight_prior=0.7,null" "decodability.kwargs.beta_pM_unif=1.7,1.9" "decodability.kwargs.beta_HMlZ=1.5,1.4"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs  -m  >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi