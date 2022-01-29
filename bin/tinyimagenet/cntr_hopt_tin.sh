#!/usr/bin/env bash

experiment=$prfx"cntr_hopt_tin"
notes="
**Goal**: hyperparameter tuning for contrastive on tinyimagenet.
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
representor=cntr
scheduler_issl.kwargs.base.is_warmup_lr=True
data@data_repr=tinyimagenet
scheduler@scheduler_issl=warm_unifmultistep
checkpoint@checkpoint_repr=bestTrainLoss
+trainer.limit_val_batches=0
++data_repr.kwargs.val_size=2
data_repr.kwargs.is_force_all_train=True
optimizer@optimizer_issl=AdamW
data_repr.kwargs.batch_size=512
encoder.kwargs.arch_kwargs.is_no_linear=True
encoder.z_shape=512
optimizer_issl.kwargs.lr=2e-3
scheduler_issl.kwargs.base.warmup_epochs=0.1
scheduler_issl.kwargs.UniformMultiStepLR.decay_per_step=5
decodability.kwargs.temperature=0.07
encoder.is_relu_Z=True
optimizer_issl.kwargs.weight_decay=5e-6
encoder.batchnorm_mode=pred
decodability.kwargs.is_self_contrastive=symmetric
regularizer=none
timeout=$time
"


# every arguments that you are sweeping over
kwargs_multi="
seed=1,2,3
trainer.max_epochs=300,1000
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "" # "decodability.kwargs.is_self_contrastive=no" "regularizer=none"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 10

  done
fi

wait

# for representor
python utils/aggregate.py \
       experiment=$experiment  \
       agg_mode=[summarize_metrics]
