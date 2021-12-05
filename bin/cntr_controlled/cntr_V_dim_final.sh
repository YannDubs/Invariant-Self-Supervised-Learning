#!/usr/bin/env bash

experiment=$prfx"cntr_V_dim_final"
notes="
**Goal**: figure showing effect of predictive family on the necessary dimensionality.
"

# MAYBE RERUN this with 250A

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
data_pred.all_data=[data_repr_agg16]
predictor=pytorch
data_repr.kwargs.val_size=2
+data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
representor=cntr
timeout=$time
"

# every arguments that you are sweeping over
kwargs_multi="
encoder.z_shape=2,5,10,20
regularizer=huber
representor.loss.beta=1e-3
seed=1,2,3
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in   "representor=cntr_mlpS architecture@predictor=mlp_h1024_l1" "representor=cntr architecture@predictor=linear"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 3

  done
fi


wait


python utils/aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       "+col_val_subset.beta=[1e-3]" \
       patterns.representor=null \
       +kwargs.pretty_renamer.Cntr="Linear" \
       +kwargs.pretty_renamer.Cntr_Mlps="MLP" \
       +plot_scatter_lines.x="zdim" \
       +plot_scatter_lines.y="test/pred/acc_agg_min" \
       +plot_scatter_lines.filename="lines_acc_vs_dim_gs" \
       +plot_scatter_lines.hue="repr" \
       +plot_scatter_lines.style="repr" \
       +plot_scatter_lines.set_kwargs.xticks=[0,5,10,15,20] \
       +plot_scatter_lines.logbase_x=1 \
       +plot_scatter_lines.legend_out=False \
       "+plot_scatter_lines.hue_order=[Cntr,Cntr_Mlps]" \
       "+plot_scatter_lines.style_order=[Cntr,Cntr_Mlps]" \
       agg_mode=[plot_scatter_lines]

python utils/aggregate.py \
       experiment=$experiment  \
       agg_mode=[summarize_metrics]