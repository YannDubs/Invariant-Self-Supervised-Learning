#!/usr/bin/env bash

experiment=$prfx"whitening_tin"
notes="
**Goal**: ensure that you can replicate the whitening paper for tinyimagenet with standard contrastive learning.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

time=4320

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
+logger.wandb_kwargs.project=tinyimagenet
architecture@encoder=resnet18
architecture@online_evaluator=linear
downstream_task.all_tasks=[sklogistic_datarepr,sklogistic_datarepr001test,sklogistic_datarepr001,sklogistic_datarepragg]
++data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
representor=std_cntr
decodability.kwargs.temperature=0.5
data_repr.kwargs.batch_size=512
scheduler_issl.kwargs.base.is_warmup_lr=True
decodability.kwargs.projector_kwargs.hid_dim=1024
decodability.kwargs.projector_kwargs.n_hid_layers=1
encoder.z_shape=512
data@data_repr=tinyimagenet
checkpoint@checkpoint_repr=bestTrainLoss
+trainer.limit_val_batches=0
++data_repr.kwargs.val_size=2
optimizer@optimizer_issl=AdamW
timeout=$time
"

# the only differences with whitening are optimization stuff: scheduler, decay, lr, optimizer,

# every arguments that you are sweeping over
kwargs_multi="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
monitor_direction=[maximize]
monitor_return=[pred/tinyimagenet/accuracy_score]
hydra.sweeper.n_trials=5
hydra.sweeper.n_jobs=5
hydra.sweeper.study_name=v2
decodability.kwargs.projector_kwargs.out_shape=64
optimizer_issl.kwargs.lr=tag(log,interval(1e-3,5e-3))
optimizer_issl.kwargs.weight_decay=tag(log,interval(5e-7,5e-6))
scheduler@scheduler_issl=warm_unifmultistep100,slowwarm_unifmultistep25,warm_unifmultistep125
seed=1,2,3,4,5,6,7,8,9
trainer.max_epochs=1000
"
# to replicate should add 1000 epochs and warm_unifmultistep125


# difference for gen: linear resnet / augmentations / larger dim


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 10

  done
fi