#!/usr/bin/env bash

experiment=$prfx"overfit_cifar10_exact"
notes="
**Goal**: understand how to get good results on cifar.
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
data_pred.all_data=[data_repr]
predictor=sk_logistic
++data_repr.kwargs.val_size=2
++data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
data@data_repr=cifar10
trainer.max_epochs=200
representor=exact_noA
data_repr.kwargs.batch_size=256
scheduler_issl.kwargs.base.is_warmup_lr=True
scheduler@scheduler_issl=unifmultistep1000
timeout=$time
"



# every arguments that you are sweeping over
kwargs_multi="
"

# difference for gen: linear resnet / augmentations / larger dim


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "architecture@encoder=resnet50"  "optimizer_issl.kwargs.lr=1e-3,3e-3"  "scheduler_issl.kwargs.base.is_warmup_lr=False"  "representor=exact_stdA,exact,exact_1000A"
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


python utils/aggregate.py \
       experiment=$experiment  \
       +col_val_subset.datapred=["stl10_agg"] \
       kwargs.prfx="agg_" \
       agg_mode=[summarize_metrics]