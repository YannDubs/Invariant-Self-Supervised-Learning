#!/usr/bin/env bash

experiment=$prfx"cntr_reg_dim"
notes="
**Goal**: effect of using regularizer for exact on dim.
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
downstream_task.all_tasks=[sklogistic_datarepragg16]
representor=cntr
timeout=$time
"


kwargs_multi="
seed=1,2,3
encoder.z_shape=10,16,32,64,128,256
"



if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in   "regularizer=none representor.loss.beta=0 trainer.max_epochs=50,100,200" "regularizer=huber representor.loss.beta=1e-3 trainer.max_epochs=50,100"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 3

  done
fi

wait


python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       +kwargs.pretty_renamer.Huber='str(True)' \
       +kwargs.pretty_renamer.None='str(False)' \
       +collect_data.params_to_add.epochs="trainer.max_epochs" \
       +collect_data.params_to_add.regularize="regularizer.name" \
       +plot_scatter_lines.x="zdim" \
       +plot_scatter_lines.y="test/pred/accuracy_score_agg_min" \
       +plot_scatter_lines.filename="lines_acc_vs_dim_reg" \
       +plot_scatter_lines.hue="regularize" \
       +plot_scatter_lines.style="epochs" \
       +plot_scatter_lines.logbase_x=2 \
       +plot_scatter_lines.legend_out=True \
       agg_mode=[plot_scatter_lines] \
       $add_kwargs

python utils/aggregate.py \
       experiment=$experiment  \
       agg_mode=[summarize_metrics] \
       $add_kwargs