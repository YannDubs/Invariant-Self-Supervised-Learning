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
downstream_task.all_tasks=[sklogistic_datarepr,sklogistic_datarepr001test]
++data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
representor=slfdstl_prior
scheduler_issl.kwargs.base.is_warmup_lr=True
data@data_repr=cifar10
timeout=$time
"


# every arguments that you are sweeping over
kwargs_multi="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
monitor_direction=[maximize]
monitor_return=[test/pred/cifar10/accuracy_score]
hydra.sweeper.n_trials=25
hydra.sweeper.n_jobs=25
trainer.max_epochs=100,200,300,500,1000
optimizer@optimizer_issl=Adam,AdamW
optimizer_issl.kwargs.lr=tag(log,interval(3e-4,1e-2))
optimizer_issl.kwargs.weight_decay=tag(log,interval(1e-8,1e-5))
scheduler@scheduler_issl=warm_unifmultistep125,whitening,warm_unifmultistep100,slowwarm_unifmultistep25,warm_unifmultistep25,warm_unifmultistep9
seed=1,2,3,4,5,6,7,8,9
encoder.z_shape=1024,2048,4096,8192
regularizer=huber,none
representor.loss.beta=tag(log,interval(1e-8,1e-4))
decodability.kwargs.ema_weight_prior=null,0.9
decodability.kwargs.n_Mx=100,1000,10000,100000
decodability.kwargs.beta_pM_unif=tag(log,interval(1e0,3e1))
decodability.kwargs.projector_kwargs.architecture=linear,mlp
decodability.kwargs.is_symmetrized=True,False
decodability.kwargs.is_ema=True,False
decodability.kwargs.is_stop_grad=True,False
data_repr.kwargs.batch_size=128,256,512
"
# is_ema and is_stop_grad are just to test that actually worst in practice


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "checkpoint@checkpoint_repr=bestValLoss" "checkpoint@checkpoint_repr=bestTrainLoss +trainer.limit_val_batches=0 ++data_repr.kwargs.val_size=2"
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
