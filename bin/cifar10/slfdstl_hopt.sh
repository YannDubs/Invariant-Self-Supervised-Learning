#!/usr/bin/env bash

experiment=$prfx"slfdst_hopt"
notes="
**Goal**: hyperparameter tuning for selfdistillation on cifar10.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
+logger.wandb_kwargs.project=cifar10
architecture@encoder=resnet18
architecture@online_evaluator=linear
downstream_task.all_tasks=[sklogistic_datarepr,sklogistic_datarepr001test,sklogistic_datarepr001]
++data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
representor=slfdstl_prior
scheduler_issl.kwargs.base.is_warmup_lr=True
checkpoint@checkpoint_repr=bestTrainLoss
scheduler@scheduler_issl=warm_unifmultistep
data@data_repr=cifar10
+trainer.limit_val_batches=0
++data_repr.kwargs.val_size=2
optimizer@optimizer_issl=AdamW
timeout=$time
"


# every arguments that you are sweeping over
kwargs_multi="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
monitor_direction=[maximize]
monitor_return=[pred/cifar10/accuracy_score]
hydra.sweeper.n_trials=20
hydra.sweeper.n_jobs=20
hydra.sweeper.study_name=v3
optimizer_issl.kwargs.lr=tag(log,interval(1e-3,5e-3))
optimizer_issl.kwargs.weight_decay=tag(log,interval(5e-7,7e-6))
scheduler_issl.kwargs.UniformMultiStepLR.decay_per_step=shuffle(range(3,8))
scheduler_issl.kwargs.base.warmup_epochs=interval(0,0.3)
seed=1,2,3,4,5,6,7,8,9
encoder.z_shape=1024,2048,4096
regularizer=huber,cosine,none
representor.loss.beta=tag(log,interval(3e-8,2e-5))
decodability.kwargs.ema_weight_prior=null,0.7
decodability.kwargs.out_dim=tag(log,int(interval(100,3000)))
decodability.kwargs.beta_pM_unif=tag(log,interval(1,3))
decodability.kwargs.projector_kwargs.architecture=linear,mlp
decodability.kwargs.is_symmetric_loss=True,False
decodability.kwargs.is_symmetric_KL_H=True
data_repr.kwargs.batch_size=128,256,512
encoder.is_normalize_Z=True,False
encoder.is_relu_Z=True,False
encoder.is_batchnorm_Z=True,False
trainer.max_epochs=200
"

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
       $col_val_subset \
       agg_mode=[summarize_metrics]
