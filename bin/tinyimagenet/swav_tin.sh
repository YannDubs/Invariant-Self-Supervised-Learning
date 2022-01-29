#!/usr/bin/env bash

experiment=$prfx"swav_tin"
notes="
**Goal**: ensure that you can replicate simsiam on tinyimagenet.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

time=5760

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
+logger.wandb_kwargs.project=tinyimagenet
architecture@encoder=resnet18
architecture@online_evaluator=linear
downstream_task.all_tasks=[sklogistic_datarepr,sklogistic_encgen,sklogistic_predgen]
++data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
representor=slfdstl_swav
data_repr.kwargs.batch_size=512
scheduler_issl.kwargs.base.is_warmup_lr=True
encoder.z_shape=512
encoder.kwargs.arch_kwargs.is_no_linear=True
data@data_repr=tinyimagenet
data_repr.kwargs.is_force_all_train=True
checkpoint@checkpoint_repr=bestTrainLoss
+trainer.limit_val_batches=0
++data_repr.kwargs.val_size=2
optimizer@optimizer_issl=AdamW
scheduler@scheduler_issl=warm_unifmultistep
optimizer_issl.kwargs.weight_decay=1e-6
optimizer_issl.kwargs.lr=2e-3
scheduler_issl.kwargs.base.warmup_epochs=0.1
decodability.kwargs.epoch_queue_starts=5
decodability.kwargs.n_Mx=500
decodability.kwargs.temperature=0.07
scheduler_issl.kwargs.UniformMultiStepLR.decay_per_step=5
timeout=$time
"

# the only differences with whitening are optimization stuff: scheduler, decay, lr, optimizer,

# every arguments that you are sweeping over
kwargs_multi="
seed=1
trainer.max_epochs=1000
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 10

  done
fi