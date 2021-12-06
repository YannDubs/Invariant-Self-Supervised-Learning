#!/usr/bin/env bash

experiment=$prfx"cntrlld_reg"
notes="
**Goal**: effect of using regularizer for exact.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
+logger.wandb_kwargs.project=exact_controlled
checkpoint@checkpoint_repr=bestTrainLoss
architecture@encoder=resnet18
architecture@online_evaluator=linear
data@data_repr=cifar10
data_repr.kwargs.val_size=2
+data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
downstream_task.all_tasks=[sklogistic_datarepragg16,sklogistic_datarepr1000agg16,sklogistic_datarepr1000agg16test]
representor=exact
timeout=$time
"


# every arguments that you are sweeping over
kwargs_multi="
seed=1
trainer.max_epochs=50,100,200,500,1000
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "regularizer=l2Mx representor.loss.beta=1e-1" "regularizer=none representor.loss.beta=0"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 3

  done
fi

wait


python utils/aggregate.py \
       experiment=$experiment  \
       "+col_val_subset.data_pred=[data_repr_agg16]" \
       patterns.representor=null \
       +collect_data.params_to_add.Epochs="trainer.max_epochs" \
       +plot_scatter_lines.x="Epochs" \
       +plot_scatter_lines.y="train/pred/accuracy_score_agg_min" \
       +plot_scatter_lines.filename="lines_epochs_vs_acc_min_tr" \
       +plot_scatter_lines.hue="beta" \
       +plot_scatter_lines.style="beta" \
       +plot_scatter_lines.logbase_x=1 \
       +plot_scatter_lines.legend_out=False \
       agg_mode=[plot_scatter_lines]

python utils/aggregate.py \
       experiment=$experiment  \
       "+col_val_subset.data_pred=[data_repr_agg16]" \
       patterns.representor=null \
       +collect_data.params_to_add.Epochs="trainer.max_epochs" \
       +plot_scatter_lines.x="Epochs" \
       +plot_scatter_lines.y="test/pred/accuracy_score_agg_min" \
       +plot_scatter_lines.filename="lines_epochs_vs_acc_min" \
       +plot_scatter_lines.hue="beta" \
       +plot_scatter_lines.style="beta" \
       +plot_scatter_lines.logbase_x=1 \
       +plot_scatter_lines.legend_out=False \
       agg_mode=[plot_scatter_lines]

python utils/aggregate.py \
       experiment=$experiment  \
       "+col_val_subset.data_pred=[data_repr_agg16]" \
       patterns.representor=null \
       +collect_data.params_to_add.Epochs="trainer.max_epochs" \
       +plot_scatter_lines.x="Epochs" \
       +plot_scatter_lines.y="train/pred/accuracy_score" \
       +plot_scatter_lines.filename="lines_epochs_vs_acc_tr" \
       +plot_scatter_lines.hue="beta" \
       +plot_scatter_lines.style="beta" \
       +plot_scatter_lines.logbase_x=1 \
       +plot_scatter_lines.legend_out=False \
       agg_mode=[plot_scatter_lines]

python utils/aggregate.py \
       experiment=$experiment  \
       "+col_val_subset.data_pred=[data_repr_agg16]" \
       patterns.representor=null \
       +collect_data.params_to_add.Epochs="trainer.max_epochs" \
       +plot_scatter_lines.x="Epochs" \
       +plot_scatter_lines.y="test/pred/accuracy_score" \
       +plot_scatter_lines.filename="lines_epochs_vs_acc" \
       +plot_scatter_lines.hue="beta" \
       +plot_scatter_lines.style="beta" \
       +plot_scatter_lines.logbase_x=1 \
       +plot_scatter_lines.legend_out=False \
       agg_mode=[plot_scatter_lines]

python utils/aggregate.py \
       experiment=$experiment  \
       "+col_val_subset.data_pred=[data_repr_1000_agg16]" \
       patterns.representor=null \
       +collect_data.params_to_add.Epochs="trainer.max_epochs" \
       +plot_scatter_lines.x="Epochs" \
       +plot_scatter_lines.y="test/pred/accuracy_score" \
       +plot_scatter_lines.filename="lines_epochs_vs_acc_mini_tr" \
       +plot_scatter_lines.hue="beta" \
       +plot_scatter_lines.style="beta" \
       +plot_scatter_lines.logbase_x=1 \
       +plot_scatter_lines.legend_out=False \
       agg_mode=[plot_scatter_lines]

python utils/aggregate.py \
       experiment=$experiment  \
       "+col_val_subset.data_pred=[data_repr_1000_agg16]" \
       patterns.representor=null \
       +collect_data.params_to_add.Epochs="trainer.max_epochs" \
       +plot_scatter_lines.x="Epochs" \
       +plot_scatter_lines.y="test/pred/accuracy_score_agg_min" \
       +plot_scatter_lines.filename="lines_epochs_vs_acc_min_mini_tr" \
       +plot_scatter_lines.hue="beta" \
       +plot_scatter_lines.style="beta" \
       +plot_scatter_lines.logbase_x=1 \
       +plot_scatter_lines.legend_out=False \
       agg_mode=[plot_scatter_lines]

python utils/aggregate.py \
       experiment=$experiment  \
       "+col_val_subset.data_pred=[data_repr_1000_agg16_test]" \
       patterns.representor=null \
       +collect_data.params_to_add.Epochs="trainer.max_epochs" \
       +plot_scatter_lines.x="Epochs" \
       +plot_scatter_lines.y="test/pred/accuracy_score" \
       +plot_scatter_lines.filename="lines_epochs_vs_acc_mini" \
       +plot_scatter_lines.hue="beta" \
       +plot_scatter_lines.style="beta" \
       +plot_scatter_lines.logbase_x=1 \
       +plot_scatter_lines.legend_out=False \
       agg_mode=[plot_scatter_lines]

python utils/aggregate.py \
       experiment=$experiment  \
       "+col_val_subset.data_pred=[data_repr_1000_agg16_test]" \
       patterns.representor=null \
       +collect_data.params_to_add.Epochs="trainer.max_epochs" \
       +plot_scatter_lines.x="Epochs" \
       +plot_scatter_lines.y="test/pred/accuracy_score_agg_min" \
       +plot_scatter_lines.filename="lines_epochs_vs_acc_min_mini" \
       +plot_scatter_lines.hue="beta" \
       +plot_scatter_lines.style="beta" \
       +plot_scatter_lines.logbase_x=1 \
       +plot_scatter_lines.legend_out=False \
       agg_mode=[plot_scatter_lines]

python utils/aggregate.py \
       experiment=$experiment  \
       agg_mode=[summarize_metrics]
