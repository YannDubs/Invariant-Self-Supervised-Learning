#!/usr/bin/env bash

experiment="dissl_ema"
notes="
**Goal**: hyperparameter tuning without ema
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10000

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
seed=2
timeout=$time
representor=dstl
downstream_task.all_tasks=[torchlogisticw1e-4_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-6_datarepr,torchlogisticw1e-5b2048e300_datarepr]
++decodability.kwargs.projector_kwargs.n_hid_layers=1
++decodability.kwargs.projector_kwargs.hid_dim=1024
"

kwargs_multi="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
monitor_direction=[maximize]
monitor_return=[pred/torchlogisticw1e-6_datarepr/acc]
hydra.sweeper.study_name=new1
hydra.sweeper.n_trials=30
hydra.sweeper.n_jobs=30
decodability.kwargs.ema_weight_prior=0.5,0.7,0.9
decodability.kwargs.beta_pM_unif=interval(1.7,2.3)
decodability.kwargs.beta_HMlZ=interval(1.3,1.8)
decodability.kwargs.out_dim=8192,16384
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
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
