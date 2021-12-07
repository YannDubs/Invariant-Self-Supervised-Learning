#!/usr/bin/env bash

experiment=$prfx"cntr_aug_samples_agg_final"
notes="
**Goal**: figure showing effect of augmentations on the necessary # samples.
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
downstream_task.all_tasks=[sklogistic_datarepr10agg16test,sklogistic_datarepr100agg16test,sklogistic_datarepr1000agg16test,sklogistic_datarepr45000agg16test]
data_repr.kwargs.val_size=2
+data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
timeout=$time
"

# every arguments that you are sweeping over
kwargs_multi="
representor=cntr,cntr_1000A,cntr_1000A_shuffle,cntr_stdA
regularizer=huber
representor.loss.beta=1e-3
seed=1,2,3
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
       patterns.representor=null \
       "+col_val_subset.beta=[1e-3]" \
       +kwargs.pretty_renamer.Cntr_100A="Finer: 100" \
       +kwargs.pretty_renamer.Cntr_1000A_Shuffle="Not Sufficient" \
       +kwargs.pretty_renamer.Cntr_1000A="Finer: 1000" \
       +kwargs.pretty_renamer.Cntr_10000A="Finer: 10000" \
       +kwargs.pretty_renamer.Cntr_Stda="Standard" \
       +kwargs.pretty_renamer.Cntr="Minimal: 10" \
       +kwargs.pretty_renamer.Repr="Augmentation" \
       +collect_data.params_to_add.n_samples="data.kwargs.subset_train_size" \
       +plot_scatter_lines.x="n_samples" \
       +plot_scatter_lines.y="test/pred/accuracy_score_agg_min" \
       +plot_scatter_lines.filename="lines_acc_vs_samples_agg_final" \
       +plot_scatter_lines.hue="repr" \
       +plot_scatter_lines.style="repr" \
       +plot_scatter_lines.logbase_x=10 \
       +plot_scatter_lines.legend_out=True \
        "+plot_scatter_lines.hue_order=[Cntr,Cntr_1000A,Cntr_Stda,Cntr_1000A_Shuffle]" \
       "+plot_scatter_lines.style_order=[Cntr,Cntr_1000A,Cntr_Stda,Cntr_1000A_Shuffle]" \
       agg_mode=[plot_scatter_lines]


python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       "+col_val_subset.beta=[1e-3]" \
       +kwargs.pretty_renamer.Cntr_100A="Finer: 100" \
       +kwargs.pretty_renamer.Cntr_1000A_Shuffle="Not Sufficient" \
       +kwargs.pretty_renamer.Cntr_1000A="Finer: 1000" \
       +kwargs.pretty_renamer.Cntr_10000A="Finer: 10000" \
       +kwargs.pretty_renamer.Cntr_Stda="Standard" \
       +kwargs.pretty_renamer.Cntr="Minimal: 10" \
       +kwargs.pretty_renamer.Repr="Augmentation" \
       +collect_data.params_to_add.n_samples="data.kwargs.subset_train_size" \
       +plot_scatter_lines.x="n_samples" \
       +plot_scatter_lines.y="test/pred/accuracy_score" \
       +plot_scatter_lines.filename="lines_acc_vs_samples_final" \
       +plot_scatter_lines.hue="repr" \
       +plot_scatter_lines.style="repr" \
       +plot_scatter_lines.logbase_x=10 \
       +plot_scatter_lines.legend_out=True \
       "+plot_scatter_lines.hue_order=[Cntr,Cntr_1000A,Cntr_Stda,Cntr_1000A_Shuffle]" \
       "+plot_scatter_lines.style_order=[Cntr,Cntr_1000A,Cntr_Stda,Cntr_1000A_Shuffle]" \
       agg_mode=[plot_scatter_lines]

python utils/aggregate.py \
       experiment=$experiment  \
       agg_mode=[summarize_metrics]