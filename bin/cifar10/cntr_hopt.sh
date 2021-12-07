#!/usr/bin/env bash

experiment=$prfx"cntr_hopt"
notes="
**Goal**: hyperparameter tuning for contrastive.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
+logger.wandb_kwargs.project=cifar10
checkpoint@checkpoint_repr=bestTrainLoss
architecture@encoder=resnet18
architecture@online_evaluator=linear
downstream_task.all_tasks=[sklogistic_datarepragg16]
++data_repr.kwargs.val_size=2
++data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
representor=cntr_stdA
decodability.kwargs.temperature=0.5
data_repr.kwargs.batch_size=512
optimizer_issl.kwargs.weight_decay=1e-6
scheduler_issl.kwargs.base.is_warmup_lr=True
encoder.z_shape=1024
trainer.max_epochs=200
data@data_repr=cifar10
decodability.kwargs.predictor_kwargs.out_shape=64
timeout=$time
"


# every arguments that you are sweeping over
kwargs_multi="
optimizer_issl.kwargs.lr=3e-3
scheduler@scheduler_issl=warm_unifmultistep25
"

# weight decay can probably be decreased


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "encoder.z_shape=2048"  "" "optimizer_issl.kwargs.weight_decay=0,1e-8,1e-7" "data_repr.kwargs.batch_size=1024" "decodability.kwargs.is_pred_proj_same=True"   "regularizer=huber representor.loss.beta=1e-6" "scheduler@scheduler_issl=cosine" "decodability.kwargs.temperature=0.3"
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
