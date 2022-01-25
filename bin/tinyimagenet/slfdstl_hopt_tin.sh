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
encoder.is_relu_Z=True
encoder.batchnorm_mode=pred
decodability.kwargs.projector_kwargs.architecture=linear
encoder.z_shape=4096
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
hydra.sweeper.study_name=v2
optimizer_issl.kwargs.lr=1e-3,2e-3
optimizer_issl.kwargs.weight_decay=tag(log,interval(1e-6,5e-6))
scheduler_issl.kwargs.UniformMultiStepLR.decay_per_step=shuffle(range(5,7))
scheduler_issl.kwargs.base.warmup_epochs=interval(0,0.2)
seed=1,2,3,4,5,6,7,8,9
regularizer=huber,none,cosine
representor.loss.beta=tag(log,interval(1e-6,4e-6))
decodability.kwargs.out_dim=500,1000
decodability.kwargs.beta_pM_unif=tag(log,interval(1.5,2.5))
decodability.kwargs.freeze_predproj_epochs=0,1
trainer.max_epochs=300
"
# try encoder.kwargs.arch_kwargs.is_no_linear=True


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "" "encoder.kwargs.arch_kwargs.is_no_linear=True encoder.z_shape=512" "encoder.is_relu_Z=False encoder.batchnorm_mode=null encoder.kwargs.arch_kwargs.is_no_linear=True encoder.z_shape=512"
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
