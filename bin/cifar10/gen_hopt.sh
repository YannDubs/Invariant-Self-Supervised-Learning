#!/usr/bin/env bash

experiment=$prfx"gen_hopt"
notes="
**Goal**: hyperparameter tuning for generative on cifar10.
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
representor=gen_stdA_clfresnet
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
monitor_return=[pred/cifar10/accuracy_score]
hydra.sweeper.study_name=v2
hydra.sweeper.n_trials=10
hydra.sweeper.n_jobs=10
optimizer@optimizer_issl=Adam,AdamW
optimizer_issl.kwargs.lr=tag(log,interval(1e-4,3e-3))
optimizer_issl.kwargs.weight_decay=tag(log,interval(1e-8,5e-6))
scheduler@scheduler_issl=warm_unifmultistep125,whitening,warm_unifmultistep100,slowwarm_unifmultistep25,warm_unifmultistep25,warm_unifmultistep9
seed=1,2,3,4,5,6,7,8,9
encoder.z_shape=512,1024,2048
regularizer=huber,cosine,rmse,none
representor.loss.beta=tag(log,interval(1e-7,1e-3))
decodability.kwargs.predecode_n_Mx=tag(log,int(interval(50,10000)))
decodability.kwargs.softmax_kwargs.temperature_mode='anneal','train','constant'
decodability.kwargs.softmax_kwargs.temperature=tag(log,interval(0.5,5))
data_repr.kwargs.batch_size=128,256,512,1024
encoder.is_normalize_Z=True,False
trainer.max_epochs=200
"


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
