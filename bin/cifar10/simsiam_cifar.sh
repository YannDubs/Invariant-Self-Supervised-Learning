#!/usr/bin/env bash

experiment=$prfx"simsiam_cifar"
notes="
**Goal**: ensure that you can replicate the simsiam paper for cifar.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
+logger.wandb_kwargs.project=cifar10
architecture@encoder=resnet18
architecture@online_evaluator=linear
downstream_task.all_tasks=[sklogistic_datarepr,sklogistic_datarepr001test]
++data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
representor=slfdstl_simsiam
data_repr.kwargs.batch_size=512
scheduler_issl.kwargs.base.is_warmup_lr=True
encoder.z_shape=512
data@data_repr=cifar10
timeout=$time
"

# the only differences with whitening are optimization stuff: scheduler, decay, lr, optimizer,

# every arguments that you are sweeping over
kwargs_multi="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
monitor_direction=[maximize]
monitor_return=[test/pred/cifar10/accuracy_score]
hydra.sweeper.n_trials=10
hydra.sweeper.n_jobs=3
optimizer@optimizer_issl=Adam,AdamW
optimizer_issl.kwargs.lr=tag(log,interval(3e-4,1e-2))
optimizer_issl.kwargs.weight_decay=tag(log,interval(1e-8,1e-5))
scheduler@scheduler_issl=warm_unifmultistep125,warm_unifmultistep100,slowwarm_unifmultistep25,warm_unifmultistep25,warm_unifmultistep9,cosine
seed=1,2,3,4,5,6,7,8,9
trainer.max_epochs=200,1000
"


# difference for gen: linear resnet / augmentations / larger dim


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "checkpoint@checkpoint_repr=bestTrainLoss +trainer.limit_val_batches=0 ++data_repr.kwargs.val_size=2" #"checkpoint@checkpoint_repr=bestValLoss"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 10

  done
fi
