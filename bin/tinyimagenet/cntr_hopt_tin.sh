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
decodability.kwargs.temperature=0.07
encoder.is_relu_Z=True
encoder.batchnorm_mode=pred
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
hydra.sweeper.study_name=v3
optimizer_issl.kwargs.lr=tag(log,interval(2e-3,4e-3))
optimizer_issl.kwargs.weight_decay=tag(log,interval(2e-6,1e-5))
scheduler_issl.kwargs.UniformMultiStepLR.decay_per_step=shuffle(range(4,8))
scheduler_issl.kwargs.base.warmup_epochs=interval(0,0.2)
seed=1,2,3,4,5,6,7,8,9
encoder.z_shape=1024,2048
regularizer=huber,none,cosine
representor.loss.beta=tag(log,interval(1e-6,3e-5))
decodability.kwargs.is_self_contrastive=no,symmetric
encoder.is_normalize_Z=True,False
trainer.max_epochs=300
"
# high temperature is better for sample efficiency but low one is better for decodability
# normalize Z is good for sample efficiency but maybe slightly worst for general ?
# try encoder.kwargs.arch_kwargs.is_no_linear=True


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
