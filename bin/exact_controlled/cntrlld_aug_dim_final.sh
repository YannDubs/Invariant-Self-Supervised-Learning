#!/usr/bin/env bash

experiment=$prfx"cntrlld_aug_dim_final"
notes="
**Goal**: figure showing effect of augmentations on the necessary dimensionality.
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
data_pred.all_data=[data_repr_agg16]
predictor=sk_logistic
data_repr.kwargs.val_size=2
+data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
timeout=$time
"


# every arguments that you are sweeping over
kwargs_multi="
representor=exact,exact_100A,exact_1000A,exact_1000A_shuffle,exact_stdA,exact_noA,exact_coarserA
encoder.z_shape=5,10,100,1000,4000
regularizer=l2Mx
representor.loss.beta=1e-1
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


# test -> tesst and train -> test is quick way to remove "train" in plot
python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       "+col_val_subset.reg=[l2Mx]" \
       "+col_val_subset.repr=[exact,exact_100A,exact_1000A_shuffle,exact_stdA,exact_noA,exact_coarserA]" \
       "+col_val_subset.zdim=[5,10,100,1000]" \
       +kwargs.pretty_renamer.Exact_100A="Finer: 100" \
       +kwargs.pretty_renamer.Exact_1000A_Shuffle="Not Sufficient" \
       +kwargs.pretty_renamer.Exact_1000A="Finer: 1000" \
       +kwargs.pretty_renamer.Exact_Stda="Standard" \
       +kwargs.pretty_renamer.Exact_Noa="None" \
       +kwargs.pretty_renamer.Exact_Coarsera="Coarser: 2" \
       +kwargs.pretty_renamer.Exact="Minimal: 10" \
       +kwargs.pretty_renamer.Repr="Augmentation" \
       +kwargs.pretty_renamer.Test="Tesst" \
       +kwargs.pretty_renamer.Train="Test" \
       +plot_scatter_lines.x="zdim" \
       +plot_scatter_lines.y="train/pred/accuracy_score_agg_min" \
       +plot_scatter_lines.filename="lines_acc_vs_dim_reg" \
       +plot_scatter_lines.hue="repr" \
       +plot_scatter_lines.style="repr" \
       +plot_scatter_lines.logbase_x=10 \
       +plot_scatter_lines.legend_out=True \
       "+plot_scatter_lines.hue_order=[exact,exact_100A,exact_stdA,exact_noA,exact_coarserA,exact_1000A_shuffle]" \
       "+plot_scatter_lines.style_order=[exact,exact_100A,exact_stdA,exact_noA,exact_coarserA,exact_1000A_shuffle]" \
       agg_mode=[plot_scatter_lines]


python utils/aggregate.py \
       experiment=$experiment  \
       agg_mode=[summarize_metrics]