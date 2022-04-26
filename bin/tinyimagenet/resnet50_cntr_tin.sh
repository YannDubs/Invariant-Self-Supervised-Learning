#!/usr/bin/env bash

experiment="resnet50_cntr_tin"
notes="
**Goal**: tuning cntr resnet50 on tinyimagenet.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10080

kwargs="
experiment=$experiment
++logger.wandb_kwargs.project=tinyimagenet
architecture@encoder=resnet18
architecture@online_evaluator=linear
encoder.kwargs.arch_kwargs.is_no_linear=True
data@data_repr=tinyimagenet
data_repr.kwargs.is_force_all_train=True
data_repr.kwargs.is_val_on_test=True
checkpoint@checkpoint_repr=last
optimizer@optimizer_issl=Adam
scheduler@scheduler_issl=cosine
optimizer_issl.kwargs.weight_decay=1e-6
optimizer_issl.kwargs.lr=2e-3
representor=cntr
data_repr.kwargs.batch_size=512
architecture@encoder=resnet50
downstream_task.all_tasks=[torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr,torchlogistic_datarepr]
timeout=$time
seed=1
encoder.rm_out_chan_aug=False
encoder.z_shape=2048
update_trainer_repr.max_epochs=200
"

kwargs_multi="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
monitor_direction=[maximize]
monitor_return=[pred/torchlogistic_datarepr/acc]
hydra.sweeper.study_name=new1
hydra.sweeper.n_trials=15
hydra.sweeper.n_jobs=15
decodability.kwargs.is_self_contrastive=True,False
decodability.kwargs.predictor_kwargs.out_shape=128,512
regularizer=none,etf,huber
representor.loss.beta=1e-1,1e-2
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs  -m  >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi