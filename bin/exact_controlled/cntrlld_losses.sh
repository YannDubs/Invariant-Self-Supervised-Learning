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
representor.loss.beta=0,1e-5,1e-3,1e-1
seed=1
"


kwargs_multi="
representor=exact,exact_hinge,exact_mse
regularizer=l2Mx
representor.loss.beta=1e-1
seed=1
downstream_task.all_tasks=[sklogistic_datarepragg16,skclflinear_datarepragg16,sksvm_datarepragg16]
trainer.max_epochs=1
+trainer.limit_train_batches=0
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 3

  done
fi

wait


# NEED TO implement plotting because they will be different losses
python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       +kwargs.pretty_renamer.Exact_Mse="MSE" \
       +kwargs.pretty_renamer.Exact_Hinge="Hinge" \
       +kwargs.pretty_renamer.Exact="Log loss" \
       +kwargs.pretty_renamer.Sk_Logistic="Log loss" \
       +kwargs.pretty_renamer.Sk_Clf_Linear="MSE" \
       +kwargs.pretty_renamer.Sk_Svm="Hinge" \
       +plot_heatmap.x="repr" \
       +plot_heatmap.y="pred" \
       +plot_heatmap.cols_to_agg=["seed"] \
       +plot_heatmap.metric="train/pred/acc_mean" \
       +plot_heatmap.filename="heatmap_losses" \
       +summarize_metrics.folder_col="beta" \
       agg_mode=[plot_heatmap]

python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       +kwargs.pretty_renamer.Exact_Mse="MSE" \
       +kwargs.pretty_renamer.Exact_Hinge="Hinge" \
       +kwargs.pretty_renamer.Exact="Log loss" \
       +kwargs.pretty_renamer.Sk_Logistic="Log loss" \
       +kwargs.pretty_renamer.Sk_Clf_Linear="MSE" \
       +kwargs.pretty_renamer.Sk_Svm="Hinge" \
       +plot_heatmap.x="repr" \
       +plot_heatmap.y="pred" \
       +plot_heatmap.cols_to_agg=["seed"] \
       +plot_heatmap.metric="train/pred/acc_agg_min_mean" \
       +plot_heatmap.filename="heatmap_V_losses" \
       +summarize_metrics.folder_col="beta" \
       agg_mode=[plot_heatmap]


python utils/aggregate.py \
       experiment=$experiment  \
       +summarize_metrics.folder_col="beta" \
       agg_mode=[summarize_metrics]
