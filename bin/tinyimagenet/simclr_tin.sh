#!/usr/bin/env bash

experiment=$prfx"simclr_tin"
notes="
**Goal**: ensure that you can replicate the whitening paper for tinyimagenet with simclr.
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
representor=cntr_simclr
data_repr.kwargs.batch_size=512
scheduler_issl.kwargs.base.is_warmup_lr=True
decodability.kwargs.projector_kwargs.hid_dim=1024
decodability.kwargs.projector_kwargs.n_hid_layers=1
data_repr.kwargs.is_force_all_train=True
encoder.z_shape=512
encoder.kwargs.arch_kwargs.is_no_linear=True
data@data_repr=tinyimagenet
checkpoint@checkpoint_repr=bestTrainLoss
+trainer.limit_val_batches=0
++data_repr.kwargs.val_size=2
optimizer_issl.kwargs.lr=2e-3
decodability.kwargs.temperature=0.07
optimizer@optimizer_issl=AdamW
scheduler@scheduler_issl=warm_unifmultistep
decodability.kwargs.projector_kwargs.out_shape=64
optimizer_issl.kwargs.weight_decay=1e-6
scheduler_issl.kwargs.UniformMultiStepLR.decay_per_step=4
scheduler_issl.kwargs.base.warmup_epochs=0.1
timeout=$time
"

# 3 for decay also works


# every arguments that you are sweeping over
kwargs_multi="
seed=1,2,3
trainer.max_epochs=300,1000
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 10

  done
fi
