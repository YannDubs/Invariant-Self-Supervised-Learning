#!/usr/bin/env bash

experiment="slfdst_hopt_ema_tin"
notes="
**Goal**: make sefldistillation work with ema.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh



time=4000

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
++logger.wandb_kwargs.project=tinyimagenet
architecture@encoder=resnet18
architecture@online_evaluator=linear
downstream_task.all_tasks=[sklogistic_datarepr,sklogistic_datarepr001test,sklogistic_datarepr01test,pytorch_bn_datarepr,pytorch_bn_datarepr001test,pytorch_datarepr]
++data_pred.kwargs.val_size=2
++trainer.num_sanity_val_steps=0
encoder.z_shape=512
encoder.kwargs.arch_kwargs.is_no_linear=True
data@data_repr=tinyimagenet
data_repr.kwargs.is_force_all_train=True
checkpoint@checkpoint_repr=bestTrainLoss
++trainer.limit_val_batches=0
++data_repr.kwargs.val_size=2
optimizer@optimizer_issl=AdamW
scheduler@scheduler_issl=warm_unifmultistep
optimizer_issl.kwargs.lr=2e-3
data_repr.kwargs.batch_size=256
representor=slfdstl
timeout=$time
"

kwargs_multi="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
monitor_direction=[maximize]
monitor_return=[pred/data_repr/accuracy_score]
hydra.sweeper.n_trials=5
hydra.sweeper.n_jobs=5
hydra.sweeper.study_name=v0
seed=3
trainer.max_epochs=1000
representor.loss.beta=3e-6,5e-6,1e-5
decodability.kwargs.beta_pM_unif=1.3,1.7,2.5
regularizer=huber
optimizer_issl.kwargs.weight_decay=3e-6,5e-6,1e-5
decodability.kwargs.ema_weight_prior=0.3,0.5,0.7
representor=slfdstl
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "regularizer=cosine" "decodability.kwargs.out_dim=7000,10000,15000" "decodability.kwargs.out_dim=30000,50000,80000 decodability.kwargs.projector_kwargs.bottleneck_size=30,50,100" "decodability.kwargs.out_dim=15000,30000 decodability.kwargs.projector_kwargs.bottleneck_size=100,200"
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
