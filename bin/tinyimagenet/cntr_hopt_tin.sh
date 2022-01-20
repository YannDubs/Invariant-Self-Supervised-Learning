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
downstream_task.all_tasks=[sklogistic_datarepr,sklogistic_datarepr001test,sklogistic_datarepr001,sklogistic_datarepragg]
++data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
representor=cntr_stdA
scheduler_issl.kwargs.base.is_warmup_lr=True
data@data_repr=tinyimagenet
scheduler@scheduler_issl=warm_unifmultistep
checkpoint@checkpoint_repr=bestTrainLoss
+trainer.limit_val_batches=0
++data_repr.kwargs.val_size=2
optimizer@optimizer_issl=AdamW
data_repr.kwargs.batch_size=512
timeout=$time
"


# every arguments that you are sweeping over
kwargs_multi="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
monitor_direction=[maximize]
monitor_return=[pred/tinyimagenet/accuracy_score]
hydra.sweeper.n_trials=1
hydra.sweeper.n_jobs=1
hydra.sweeper.study_name=main
optimizer_issl.kwargs.lr=tag(log,interval(1e-3,5e-3))
optimizer_issl.kwargs.weight_decay=tag(log,interval(1e-7,1e-5))
scheduler_issl.kwargs.UniformMultiStepLR.decay_per_step=shuffle(range(4,8))
scheduler_issl.kwargs.base.warmup_epochs=interval(0,0.3)
seed=1,2,3,4,5,6,7,8,9
encoder.z_shape=512,1024,2048
regularizer=huber,none
representor.loss.beta=tag(log,interval(3e-7,3e-5))
decodability.kwargs.temperature=0.07,0.1,0.3
decodability.kwargs.is_self_contrastive=yes,no,symmetric
encoder.is_normalize_Z=True,False
encoder.is_relu_Z=True,False
encoder.is_batchnorm_Z=True
encoder.batchnorm_mode=pre,pred,null
trainer.max_epochs=300
"
# only train 200 epochs to make sure not too long
# reincorporate warm_unifmultistep125 when longer epochs
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
       $col_val_subset \
       agg_mode=[summarize_metrics]
