#!/usr/bin/env bash

experiment=$prfx"cntr_V_heatmap_final"
notes="
**Goal**: figure showing effect of predictive family depending on downstream family.
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
data_pred.all_data=[data_repr_agg16]
predictor=pytorch
data_repr.kwargs.val_size=2
+data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
representor=cntr_stdA
timeout=$time
"


# every arguments that you are sweeping over
kwargs_multi="
architecture@predictor=linear,mlp_h10_l1,mlp_h2048_l2
representor=cntr_stdA,cntr_stdA_mlpnano,cntr_stdA_mlp
regularizer=huber
representor.loss.beta=1e-3,1e-1
seed=1
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 3

  done
fi

wait


# add split between yours and standard
# add renaming
python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       +kwargs.pretty_renamer.Cntr_Mlpnano="MLP --" \
       +kwargs.pretty_renamer.Cntr_Mlp="MLP ++" \
       +kwargs.pretty_renamer.Cntr="Linear" \
       +kwargs.pretty_renamer.Mlp_H2048_L2="MLP ++" \
       +kwargs.pretty_renamer.Mlp_H10_L1="MLP --" \
       +plot_heatmap.x="repr" \
       +plot_heatmap.y="pred" \
       +plot_heatmap.cols_to_agg=["seed"] \
       +plot_heatmap.metric="train/pred/acc_mean" \
       +plot_heatmap.filename="heatmap_V_reg" \
       +plot_heatmap.col="beta" \
       agg_mode=[plot_heatmap]

python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       +kwargs.pretty_renamer.Cntr_Mlpnano="MLP --" \
       +kwargs.pretty_renamer.Cntr_Mlp="MLP ++" \
       +kwargs.pretty_renamer.Cntr="Linear" \
       +kwargs.pretty_renamer.Mlp_H2048_L2="MLP ++" \
       +kwargs.pretty_renamer.Mlp_H10_L1="MLP --" \
       +plot_heatmap.x="repr" \
       +plot_heatmap.y="pred" \
       +plot_heatmap.cols_to_agg=["seed"] \
       +plot_heatmap.metric="train/pred/acc_agg_min_mean" \
       +plot_heatmap.filename="heatmap_V_min_reg" \
       +plot_heatmap.col="beta" \
       agg_mode=[plot_heatmap]

python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       "+col_val_subset.beta=[1e-3]" \
       +kwargs.pretty_renamer.Cntr_Mlpnano="MLP --" \
       +kwargs.pretty_renamer.Cntr_Mlp="MLP ++" \
       +kwargs.pretty_renamer.Cntr="Linear" \
       +kwargs.pretty_renamer.Mlp_H2048_L2="MLP ++" \
       +kwargs.pretty_renamer.Mlp_H10_L1="MLP --" \
       +plot_heatmap.x="repr" \
       +plot_heatmap.y="pred" \
       +plot_heatmap.cols_to_agg=["seed"] \
       +plot_heatmap.metric="train/pred/acc_mean" \
       +plot_heatmap.filename="heatmap_V" \
       agg_mode=[plot_heatmap]

python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       "+col_val_subset.beta=[1e-3]" \
       +kwargs.pretty_renamer.Cntr_Mlpnano="MLP --" \
       +kwargs.pretty_renamer.Cntr_Mlp="MLP ++" \
       +kwargs.pretty_renamer.Cntr="Linear" \
       +kwargs.pretty_renamer.Mlp_H2048_L2="MLP ++" \
       +kwargs.pretty_renamer.Mlp_H10_L1="MLP --" \
       +plot_heatmap.x="repr" \
       +plot_heatmap.y="pred" \
       +plot_heatmap.cols_to_agg=["seed"] \
       +plot_heatmap.metric="train/pred/acc_agg_min_mean" \
       +plot_heatmap.filename="heatmap_V_min" \
       agg_mode=[plot_heatmap]


python utils/aggregate.py \
       experiment=$experiment  \
       agg_mode=[summarize_metrics]