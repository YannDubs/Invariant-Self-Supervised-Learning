#!/usr/bin/env bash

experiment=$prfx"cntrlld_V_dim_small_final"
notes="
**Goal**: figure showing effect of predictive family on the necessary dimensionality.
"

# MAYBE RERUN this with 250A

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
data_pred.all_data=[pytorch_datarepragg16]
data_repr.kwargs.val_size=2
+data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
representor=exact
timeout=$time
"

# every arguments that you are sweeping over
kwargs_multi="
encoder.z_shape=1,2,5,10,20
regularizer=l2Mx
representor.loss.beta=1e-1
seed=1,2,3
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in   "representor=exact_mlpnano +architecture@predictor=mlp_h10_l1" "representor=exact +architecture@predictor=linear" "representor=exact_mlp +architecture@predictor=mlp_h2048_l2"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 3

  done
fi

wait

# NB accuracy_score_agg_min replaced by `acc_agg_min` because pytorch predictor instead of sklearn
# test -> tesst and train -> test is quick way to remove "train" in plot
python utils/aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       patterns.representor=null \
       +kwargs.pretty_renamer.Exact_Mlpnano="calF" \
       +kwargs.pretty_renamer.Exact_Mlp="calF ++" \
       +kwargs.pretty_renamer.Exact="calF --" \
       +kwargs.pretty_renamer.Test="Tesst" \
       +kwargs.pretty_renamer.Train="Test" \
       +plot_scatter_lines.x="zdim" \
       +plot_scatter_lines.y="train/pred/acc_agg_min" \
       +plot_scatter_lines.filename="lines_acc_vs_dim" \
       +plot_scatter_lines.hue="repr" \
       +plot_scatter_lines.style="repr" \
       +plot_scatter_lines.logbase_x=1 \
       +plot_scatter_lines.legend_out=False \
       "+plot_scatter_lines.hue_order=[Exact,Exact_Mlpnano,Exact_Mlp]" \
       "+plot_scatter_lines.style_order=[Exact,Exact_Mlpnano,Exact_Mlp]" \
       agg_mode=[plot_scatter_lines]

python utils/aggregate.py \
       experiment=$experiment  \
       agg_mode=[summarize_metrics]