#!/usr/bin/env bash

experiment=$prfx"cntrlld_permutation"
notes="
**Goal**: table showing that permuting M(X) has very little effect when doing exact ISSL.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
+logger.wandb_kwargs.project=cntr_controlled
trainer.max_epochs=100
checkpoint@checkpoint_repr=bestTrainLoss
architecture@encoder=resnet18
architecture@online_evaluator=linear
data@data_repr=cifar10
downstream_task.all_tasks=[sklogistic_datarepragg16]
data_repr.kwargs.val_size=2
+data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
representor=cntr
timeout=$time
"

# would be good to run all of them but most important is cntr
kwargs_multi="
regularizer=l2Mx
representor.loss.beta=1e-1
seed=1,2,3
representor=exact,exact_permMx
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 3

  done
fi

wait


python utils/aggregate.py \
       experiment=$experiment  \
       agg_mode=[summarize_metrics]