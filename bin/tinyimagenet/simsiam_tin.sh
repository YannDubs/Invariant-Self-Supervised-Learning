#!/usr/bin/env bash

experiment=$prfx"simsiam_tin"
notes="
**Goal**: ensure that you can replicate simsiam on tinyimagenet.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh


# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
+logger.wandb_kwargs.project=tinyimagenet
architecture@encoder=resnet18
architecture@online_evaluator=linear
downstream_task.all_tasks=[sklogistic_datarepr,sklogistic_encgen,sklogistic_predgen]
++data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
representor=slfdstl_simsiam
data_repr.kwargs.batch_size=512
scheduler_issl.kwargs.base.is_warmup_lr=True
encoder.z_shape=512
data@data_repr=tinyimagenet
data_repr.kwargs.is_force_all_train=True
checkpoint@checkpoint_repr=bestTrainLoss
+trainer.limit_val_batches=0
++data_repr.kwargs.val_size=2
optimizer@optimizer_issl=AdamW
scheduler@scheduler_issl=warm_unifmultistep
timeout=$time
"

# every arguments that you are sweeping over
kwargs_multi="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
monitor_direction=[maximize]
monitor_return=[test/pred/data_repr/accuracy_score]
hydra.sweeper.n_trials=15
hydra.sweeper.n_jobs=15
hydra.sweeper.study_name=v1
optimizer_issl.kwargs.lr=tag(log,interval(1e-3,4e-3))
optimizer_issl.kwargs.weight_decay=tag(log,interval(5e-7,5e-6))
scheduler_issl.kwargs.UniformMultiStepLR.decay_per_step=shuffle(range(3,8))
scheduler_issl.kwargs.base.warmup_epochs=interval(0,0.3)
seed=1,2,3,4,5,6,7,8,9
trainer.max_epochs=300
"


# difference for gen: linear resnet / augmentations / larger dim


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 10

  done
fi
