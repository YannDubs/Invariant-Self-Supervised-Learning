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
timeout=$time
"


# every arguments that you are sweeping over
kwargs_multi="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
monitor_direction=[maximize]
monitor_return=[test/pred/data_repr/accuracy_score]
hydra.sweeper.n_trials=5
hydra.sweeper.n_jobs=5
hydra.sweeper.study_name=v4
optimizer_issl.kwargs.weight_decay=5e-6
seed=1,2,3,4,5,6,7,8,9
regularizer=huber,none
representor.loss.beta=tag(log,interval(3e-6,1e-5))
decodability.kwargs.is_self_contrastive=symmetric
encoder.is_relu_Z=False,True
encoder.batchnorm_mode=null,pred
decodability.kwargs.temperature=0.03,0.05,0.07,0.1
trainer.max_epochs=300
"
# high temperature is better for sample efficiency but low one is better for decodability
# normalize Z is good for sample efficiency but maybe slightly worst for general ?


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
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
