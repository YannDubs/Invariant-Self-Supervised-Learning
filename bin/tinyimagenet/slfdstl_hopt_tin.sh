#!/usr/bin/env bash

experiment=$prfx"slfdst_hopt_tin"
notes="
**Goal**: hyperparameter tuning for selfdistillation on tinyimagenet.
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
representor=slfdstl
scheduler_issl.kwargs.base.is_warmup_lr=True
checkpoint@checkpoint_repr=bestTrainLoss
scheduler@scheduler_issl=warm_unifmultistep
data_repr.kwargs.is_force_all_train=True
data@data_repr=tinyimagenet
+trainer.limit_val_batches=0
++data_repr.kwargs.val_size=2
optimizer@optimizer_issl=AdamW
data_repr.kwargs.batch_size=256
decodability.kwargs.projector_kwargs.architecture=linear
encoder.z_shape=512
encoder.kwargs.arch_kwargs.is_no_linear=True
scheduler_issl.kwargs.UniformMultiStepLR.decay_per_step=5
scheduler_issl.kwargs.base.warmup_epochs=0.1
decodability.kwargs.out_dim=500
optimizer_issl.kwargs.lr=2e-3
optimizer_issl.kwargs.weight_decay=1e-6
timeout=$time
"


kwargs_multi="
seed=1,2,3
regularizer=huber
representor.loss.beta=1e-6
decodability.kwargs.beta_pM_unif=1.7
encoder.is_relu_Z=True
encoder.batchnorm_mode=pred
trainer.max_epochs=300
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
hydra.sweeper.study_name=v7
seed=1,2,3,4,5,6,7,8,9
regularizer=huber,none,cosine
representor.loss.beta=tag(log,interval(1e-6,4e-6))
decodability.kwargs.beta_pM_unif=interval(1.5,2)
encoder.is_relu_Z=False,True
encoder.batchnorm_mode=null,pred
trainer.max_epochs=300
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
hydra.sweeper.study_name=v8
seed=1,2,3,4,5,6,7,8,9
regularizer=none
decodability.kwargs.beta_pM_unif=interval(1.5,2.5)
encoder.is_relu_Z=True
encoder.batchnorm_mode=pred
decodability.kwargs.ema_weight_prior=0.5
trainer.max_epochs=300
"





if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs  -m &

    sleep 10

  done
fi

wait

# for representor
python utils/aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       agg_mode=[summarize_metrics]
