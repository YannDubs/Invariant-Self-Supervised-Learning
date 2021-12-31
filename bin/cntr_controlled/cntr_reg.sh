#!/usr/bin/env bash

experiment=$prfx"cntr_reg"
notes="
**Goal**: effect of using regularizer for contrastive.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
+logger.wandb_kwargs.project=cntr_controlled
checkpoint@checkpoint_repr=bestTrainLoss
architecture@encoder=resnet18
architecture@online_evaluator=linear
data@data_repr=cifar10
data_repr.kwargs.val_size=2
+data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
downstream_task.all_tasks=[sklogistic_datarepragg16,sklogistic_datarepr1000agg16,sklogistic_datarepr1000agg16test]
representor=cntr_stdA
timeout=$time
"


# every arguments that you are sweeping over
kwargs_multi="
seed=2,3
trainer.max_epochs=50,100,200,500,1000
"
# seed 2,3 once you found favorite beta



if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "regularizer=huber representor.loss.beta=1e-3" "regularizer=none representor.loss.beta=0"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 3

  done
fi

wait




python utils/aggregate.py \
       experiment=$experiment  \
       "+col_val_subset.datapred=[cifar10_agg16]" \
       "+col_val_subset.epochs=[50,100,200,500,1000]" \
       patterns.representor=null \
       +collect_data.params_to_add.epochs="trainer.max_epochs" \
       +collect_data.params_to_add.regularize="regularizer.name" \
       +kwargs.pretty_renamer.Test="Tesst" \
       +kwargs.pretty_renamer.Train="Test" \
       +kwargs.pretty_renamer.L2Mx='str(True)' \
       +kwargs.pretty_renamer.None='str(False)' \
       +plot_scatter_lines.x="Epochs" \
       +plot_scatter_lines.y="train/pred/accuracy_score_agg_min" \
       +plot_scatter_lines.filename="lines_epochs_vs_acc_min_tr" \
       +plot_scatter_lines.hue="regularize" \
       +plot_scatter_lines.style="regularize" \
       +plot_scatter_lines.logbase_x=null \
       +plot_scatter_lines.legend_out=True \
       agg_mode=[plot_scatter_lines]

python utils/aggregate.py \
       experiment=$experiment  \
       "+col_val_subset.datapred=[cifar10_agg16]" \
       "+col_val_subset.epochs=[50,100,200,500,1000]" \
       patterns.representor=null \
       +collect_data.params_to_add.epochs="trainer.max_epochs" \
       +collect_data.params_to_add.regularize="regularizer.name" \
       +kwargs.pretty_renamer.L2Mx='str(True)' \
       +kwargs.pretty_renamer.None='str(False)' \
       +plot_scatter_lines.x="Epochs" \
       +plot_scatter_lines.y="test/pred/accuracy_score_agg_min" \
       +plot_scatter_lines.filename="lines_epochs_vs_acc_min" \
       +plot_scatter_lines.hue="regularize" \
       +plot_scatter_lines.style="regularize" \
       +plot_scatter_lines.logbase_x=null \
       +plot_scatter_lines.legend_out=True \
       agg_mode=[plot_scatter_lines]


python utils/aggregate.py \
       experiment=$experiment  \
       "+col_val_subset.datapred=[cifar10_agg16]" \
       "+col_val_subset.epochs=[50,100,200,500,1000]" \
       patterns.representor=null \
       +collect_data.params_to_add.epochs="trainer.max_epochs" \
       +collect_data.params_to_add.regularize="regularizer.name" \
       +kwargs.pretty_renamer.Test="Tesst" \
       +kwargs.pretty_renamer.Train="Test" \
       +kwargs.pretty_renamer.L2Mx='str(True)' \
       +kwargs.pretty_renamer.None='str(False)' \
       +plot_scatter_lines.x="Epochs" \
       +plot_scatter_lines.y="train/pred/accuracy_score" \
       +plot_scatter_lines.filename="lines_epochs_vs_acc_tr" \
       +plot_scatter_lines.hue="regularize" \
       +plot_scatter_lines.style="regularize" \
       +plot_scatter_lines.logbase_x=null \
       +plot_scatter_lines.legend_out=True \
       agg_mode=[plot_scatter_lines]

python utils/aggregate.py \
       experiment=$experiment  \
       "+col_val_subset.datapred=[cifar10_agg16]" \
       "+col_val_subset.epochs=[50,100,200,500,1000]" \
       patterns.representor=null \
       +collect_data.params_to_add.epochs="trainer.max_epochs" \
       +collect_data.params_to_add.regularize="regularizer.name" \
       +kwargs.pretty_renamer.L2Mx='str(True)' \
       +kwargs.pretty_renamer.None='str(False)' \
       +plot_scatter_lines.x="Epochs" \
       +plot_scatter_lines.y="test/pred/accuracy_score" \
       +plot_scatter_lines.filename="lines_epochs_vs_acc" \
       +plot_scatter_lines.hue="regularize" \
       +plot_scatter_lines.style="regularize" \
       +plot_scatter_lines.logbase_x=null \
       +plot_scatter_lines.legend_out=True \
       agg_mode=[plot_scatter_lines]




python utils/aggregate.py \
       experiment=$experiment  \
       "+col_val_subset.datapred=[cifar10_1000_agg16]" \
       "+col_val_subset.epochs=[50,100,200,500,1000]" \
       patterns.representor=null \
       +collect_data.params_to_add.epochs="trainer.max_epochs" \
       +collect_data.params_to_add.regularize="regularizer.name" \
       +kwargs.pretty_renamer.Test="Tesst" \
       +kwargs.pretty_renamer.Train="Test" \
       +kwargs.pretty_renamer.L2Mx='str(True)' \
       +kwargs.pretty_renamer.None='str(False)' \
       +plot_scatter_lines.x="Epochs" \
       +plot_scatter_lines.y="train/pred/accuracy_score" \
       +plot_scatter_lines.filename="lines_epochs_vs_acc_mini_tr" \
       +plot_scatter_lines.hue="regularize" \
       +plot_scatter_lines.style="regularize" \
       +plot_scatter_lines.logbase_x=null \
       +plot_scatter_lines.legend_out=True \
       agg_mode=[plot_scatter_lines]

python utils/aggregate.py \
       experiment=$experiment  \
       "+col_val_subset.datapred=[cifar10_1000_agg16]" \
       "+col_val_subset.epochs=[50,100,200,500,1000]" \
       patterns.representor=null \
       +collect_data.params_to_add.epochs="trainer.max_epochs" \
       +collect_data.params_to_add.regularize="regularizer.name" \
       +kwargs.pretty_renamer.L2Mx='str(True)' \
       +kwargs.pretty_renamer.None='str(False)' \
       +plot_scatter_lines.x="Epochs" \
       +plot_scatter_lines.y="test/pred/accuracy_score_agg_min" \
       +plot_scatter_lines.filename="lines_epochs_vs_acc_min_mini_tr" \
       +plot_scatter_lines.hue="regularize" \
       +plot_scatter_lines.style="regularize" \
       +plot_scatter_lines.logbase_x=null \
       +plot_scatter_lines.legend_out=True \
       agg_mode=[plot_scatter_lines]




python utils/aggregate.py \
       experiment=$experiment  \
       "+col_val_subset.datapred=[cifar10_1000_agg16_test]" \
       "+col_val_subset.epochs=[50,100,200,500,1000]" \
       patterns.representor=null \
       +collect_data.params_to_add.epochs="trainer.max_epochs" \
       +collect_data.params_to_add.regularize="regularizer.name" \
       +kwargs.pretty_renamer.L2Mx='str(True)' \
       +kwargs.pretty_renamer.None='str(False)' \
       +plot_scatter_lines.x="Epochs" \
       +plot_scatter_lines.y="test/pred/accuracy_score_agg_min" \
       +plot_scatter_lines.filename="lines_epochs_vs_acc_min_mini" \
       +plot_scatter_lines.hue="regularize" \
       +plot_scatter_lines.style="regularize" \
       +plot_scatter_lines.logbase_x=null \
       +plot_scatter_lines.legend_out=True \
       agg_mode=[plot_scatter_lines]


python utils/aggregate.py \
       experiment=$experiment  \
       "+col_val_subset.datapred=[cifar10_1000_agg16_test]" \
       "+col_val_subset.epochs=[50,100,200,500,1000]" \
       patterns.representor=null \
       +collect_data.params_to_add.epochs="trainer.max_epochs" \
       +collect_data.params_to_add.regularize="regularizer.name" \
       +kwargs.pretty_renamer.L2Mx='str(True)' \
       +kwargs.pretty_renamer.None='str(False)' \
       +plot_scatter_lines.x="Epochs" \
       +plot_scatter_lines.y="test/pred/accuracy_score" \
       +plot_scatter_lines.filename="lines_epochs_vs_acc_mini" \
       +plot_scatter_lines.hue="regularize" \
       +plot_scatter_lines.style="regularize" \
       +plot_scatter_lines.logbase_x=null \
       +plot_scatter_lines.legend_out=True \
       agg_mode=[plot_scatter_lines]


python utils/aggregate.py \
       experiment=$experiment  \
       agg_mode=[summarize_metrics]
