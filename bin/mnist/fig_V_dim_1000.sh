#!/usr/bin/env bash

experiment=$prfx"fig_V_dim_1000"
notes="
**Goal**: figure showing effect of predictive family on the necessary dimensionality when augmentations make 1000 different equiv classes.
"

# MAYBE RERUN this with 250A

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
+logger.wandb_kwargs.project=mnist
trainer.max_epochs=50
checkpoint@checkpoint_repr=bestTrainLoss
architecture@encoder=resnet18
architecture@online_evaluator=linear
data@data_repr=mnist
data_pred.all_data=[data_repr_agg16]
predictor=pytorch
data_repr.kwargs.val_size=2
+data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
representor=cntr
timeout=$time
$add_kwargs
"


# every arguments that you are sweeping over
kwargs_multi="
encoder.z_shape=2,4,8,16,32,64,128,256,512,1024,2048
seed=1,2,3
"


# difference for gen: linear resnet / augmentations / larger dim


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in   "representor=cntr_1000A architecture@predictor=linear" "representor=cntr_1000A_mlp,cntr_1000A_stdmlp architecture@predictor=mlp_h2048_l1"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3

  done
fi

wait

# NB accuracy_score_agg_min replaced by `acc_agg_min` because pytorch predictor instead of sklearn
python utils/aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       patterns.representor=null \
       "+col_val_subset.repr=[cntr_1000A_mlp,cntr_1000A_stdmlp]" \
       +kwargs.pretty_renamer.Cntr_1000a="calF --" \
       +kwargs.pretty_renamer.cntr_10000a_Mlp="calF ++" \
       +kwargs.pretty_renamer.cntr_1000a_Stdmlp="Standard" \
       +plot_scatter_lines.x="zdim" \
       +plot_scatter_lines.y="test/pred/acc_agg_min" \
       +plot_scatter_lines.filename="lines_acc_vs_dim" \
       +plot_scatter_lines.hue="repr" \
       +plot_scatter_lines.style="repr" \
       +plot_scatter_lines.logbase_x=2 \
       +plot_scatter_lines.legend_out=False \
       agg_mode=[plot_scatter_lines]

python utils/aggregate.py \
       experiment=$experiment  \
       agg_mode=[summarize_metrics]