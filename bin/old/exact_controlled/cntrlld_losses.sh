#!/usr/bin/env bash

experiment=$prfx"cntrlld_losses"
notes="
**Goal**: effect of using different losses.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
+logger.wandb_kwargs.project=exact_controlled
trainer.max_epochs=100
checkpoint@checkpoint_repr=bestTrainLoss
architecture@encoder=resnet18
architecture@online_evaluator=linear
data@data_repr=cifar10
data_repr.kwargs.val_size=2
+data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
representor=exact
downstream_task.all_tasks=[sklogistic_datarepragg16,skclflinear_datarepragg16,sksvm_datarepragg16]
regularizer=l2Mx
timeout=$time
"


# every arguments that you are sweeping over
kwargs_multi="
representor=exact,exact_hinge,exact_mse
representor.loss.beta=1e-3
seed=1,2,3
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 3

  done
fi

wait



python utils/aggregate.py \
       experiment=$experiment  \
       +summarize_metrics.folder_col="beta" \
       "+merge_cols.losses=['train/pred/log_loss','train/pred/hinge_loss','train/pred/ridge_clf_loss']" \
       "+merge_cols.min_losses=['train/pred/log_loss_agg_min','train/pred/hinge_loss_agg_min','train/pred/ridge_clf_loss_agg_min']" \
       "+merge_cols.test_losses=['test/pred/log_loss','test/pred/hinge_loss','test/pred/ridge_clf_loss']" \
       "+merge_cols.test_min_losses=['test/pred/log_loss_agg_min','test/pred/hinge_loss_agg_min','test/pred/ridge_clf_loss_agg_min']" \
       agg_mode=[summarize_metrics]

# for plotting use:
#import pandas as pd
#
#df = pd.read_csv("summarized_metrics_predictor.csv").round(decimals=3)
#
#from IPython.display import display
#
#display(df.pivot_table(values="test_min_losses_mean",index="repr",columns="pred"))
#print()
#display(df.pivot_table(values="test_min_losses_sem",index="repr",columns="pred"))
#
#print()
#display(1-df[["repr","pred",'test/pred/accuracy_score_agg_min_mean']].set_index(["repr","pred"]))
#display(df[["repr","pred",'test/pred/accuracy_score_agg_min_sem']].set_index(["repr","pred"]))