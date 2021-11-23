#!/usr/bin/env bash

experiment=$prfx"cntrlld_aug_samples"
notes="
**Goal**: figure showing effect of augmentations on the necessary # samples.
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
data_pred.all_data=[data_repr_agg16_10,data_repr_agg16_30,data_repr_agg16_100,data_repr_agg16_1000,data_repr_agg16_10000,data_repr_agg16_50000]
predictor=sk_logistic
data_repr.kwargs.val_size=2
+data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
timeout=$time
"

# every arguments that you are sweeping over
kwargs_multi="
representor=exact,exact_250A,exact_1000A,exact_stdA,exact_noA,exact_coarserA
seed=1
"
# TODO: run seed 2,3

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
       "+col_val_subset.repr=[exact,exact_250A,exact_1000A,exact_1000A_shuffle,exact_stdA,exact_noA,exact_coarserA]" \
       patterns.representor=null \
       +kwargs.pretty_renamer.Exact_1000A_Shuffle="Not Sufficient" \
       +kwargs.pretty_renamer.Exact_1000A="Minimal --" \
       +kwargs.pretty_renamer.Exact_Stda="Standard" \
       +kwargs.pretty_renamer.Exact_Noa="None" \
       +kwargs.pretty_renamer.Exact_Coarsera="Coarser" \
       +kwargs.pretty_renamer.Exact="Minimal ++" \
       +collect_data.params_to_add.n_samples="data.kwargs.subset_train_size" \
       +plot_scatter_lines.x="n_samples" \
       +plot_scatter_lines.y="test/pred/accuracy_score" \
       +plot_scatter_lines.filename="lines_acc_vs_samples" \
       +plot_scatter_lines.hue="repr" \
       +plot_scatter_lines.style="repr" \
       +plot_scatter_lines.logbase_x=10 \
       +plot_scatter_lines.legend_out=False \
       agg_mode=[plot_scatter_lines]


  python utils/aggregate.py \
       experiment=$experiment  \
       "+col_val_subset.repr=[exact,exact_250A,exact_1000A,exact_1000A_shuffle,exact_stdA,exact_noA,exact_coarserA]" \
       patterns.representor=null \
       +kwargs.pretty_renamer.Exact_250A="Minimal" \
       +kwargs.pretty_renamer.Exact_1000A_Shuffle="Not Sufficient" \
       +kwargs.pretty_renamer.Exact_1000A="Minimal --" \
       +kwargs.pretty_renamer.Exact_Stda="Standard" \
       +kwargs.pretty_renamer.Exact_Noa="None" \
       +kwargs.pretty_renamer.Exact_Coarsera="Coarser" \
       +kwargs.pretty_renamer.Exact="Minimal ++" \
       +collect_data.params_to_add.n_samples="data.kwargs.subset_train_size" \
       +plot_scatter_lines.x="n_samples" \
       +plot_scatter_lines.y="test/pred/accuracy_score_agg_min" \
       +plot_scatter_lines.filename="lines_acc_vs_samples_agg" \
       +plot_scatter_lines.hue="repr" \
       +plot_scatter_lines.style="repr" \
       +plot_scatter_lines.logbase_x=10 \
       +plot_scatter_lines.legend_out=False \
       agg_mode=[plot_scatter_lines]


python utils/aggregate.py \
       experiment=$experiment  \
       agg_mode=[summarize_metrics]