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
downstream_task.all_tasks=[pytorch_datarepr,pytorch_datarepr001test,pytorch_datarepr01test,pytorch_bn_datarepr,pytorch_bn_datarepr01test,pytorch_bn_datarepr001test,sklogistic_datarepr,sklogistic_datarepr01test,sklogistic_datarepr001test]
data_repr.kwargs.batch_size=512
encoder.z_shape=512
encoder.kwargs.arch_kwargs.is_no_linear=True
data@data_repr=tinyimagenet
data_repr.kwargs.is_force_all_train=True
data_repr.kwargs.is_val_on_test=True
checkpoint@checkpoint_repr=bestTrainLoss
optimizer@optimizer_issl=AdamW
scheduler@scheduler_issl=warm_unifmultistep
optimizer_issl.kwargs.lr=2e-3
timeout=$time
"

kwargs_multi="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
monitor_direction=[maximize]
monitor_return=[pred/data_repr/accuracy_score]
hydra.sweeper.n_trials=20
hydra.sweeper.n_jobs=20
seed=3
hydra.sweeper.study_name=v6
trainer.max_epochs=1000
representor.loss.beta=3e-6,5e-6
decodability.kwargs.beta_pM_unif=1.7
regularizer=huber
optimizer_issl.kwargs.weight_decay=1e-6,3e-6
decodability.kwargs.ema_weight_prior=0.5,0.8,null
decodability.kwargs.out_dim=10000,20000,30000,50000
decodability.kwargs.projector_kwargs.bottleneck_size=100
representor=slfdstl
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
