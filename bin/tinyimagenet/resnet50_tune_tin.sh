#!/usr/bin/env bash

experiment="resnet50_hopt_tin"
notes="
**Goal**: tuning resnet50 on tinyimagenet.
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
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr,torchlogisticw1e-3_datarepr,torchlogisticw1e-5_datarepr001test,torchlogisticw1e-4_datarepr001test,torchlogisticw1e-3_datarepr001test,torchmlpw1e-5_datarepr,torchmlpw1e-4_datarepr,torchmlp_datarepr,sklogistic_datarepr,sklogisticreg01_datarepr,sklogisticreg001_datarepr]
encoder.kwargs.arch_kwargs.is_no_linear=True
data@data_repr=tinyimagenet
data_repr.kwargs.is_force_all_train=True
data_repr.kwargs.is_val_on_test=True
checkpoint@checkpoint_repr=last
optimizer@optimizer_issl=Adam
scheduler@scheduler_issl=cosine
optimizer_issl.kwargs.weight_decay=1e-6
optimizer_issl.kwargs.lr=2e-3
representor=dstl
architecture@encoder=resnet50
downstream_task.all_tasks=[torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
timeout=$time
encoder.kwargs.arch_kwargs.is_channel_out_dim=True
+decodability.kwargs.projector_kwargs.in_shape=2048
+encoder.kwargs.arch_kwargs.bottleneck_channel=512
seed=1
encoder.rm_out_chan_aug=True
"

kwargs_multi="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
monitor_direction=[maximize]
monitor_return=[pred/torchlogistic_datarepr/acc]
hydra.sweeper.n_trials=10
hydra.sweeper.n_jobs=10
decodability.kwargs.predictor_kwargs.bottleneck_size=256,512
decodability.kwargs.predictor_kwargs.is_train_bottleneck=False,True
regularizer=rel_l1,rel_var,none
encoder.z_shape=8192,4096,2048
update_trainer_repr.max_epochs=200,500
representor.loss.beta=1e-2,1e-1,1
decodability.kwargs.ema_weight_prior=0.7,0.9
decodability.kwargs.beta_pM_unif=1.7,1.9
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs  -m  >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi