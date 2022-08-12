#!/usr/bin/env bash

experiment=$prfx"cntr_V_heatmap_smallZ_final"
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

kwargs_multi="
architecture@predictor=linear,mlp_h32_l1
representor=cntr_stdA,cntr_stdA_mlpXS
seed=1,2,3
encoder.z_shape=10
regularizer=huber
representor.loss.beta=1e-3
"



if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in   ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 3

  done
fi

wait

python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       +kwargs.pretty_renamer.Cntr_Stda_Mlpxs="MLP" \
       +kwargs.pretty_renamer.Cntr_Stda="Linear" \
       +kwargs.pretty_renamer.Mlp_H32_L1="MLP" \
       +kwargs.pretty_renamer.Pred="Predictor" \
       +kwargs.pretty_renamer.Repr="ISSL" \
       +plot_heatmap.x="repr" \
       +plot_heatmap.y="pred" \
       +plot_heatmap.cols_to_agg=["seed"] \
       +plot_heatmap.metric="train/pred/acc_agg_min_mean" \
       +plot_heatmap.filename="heatmap_V_min_square" \
       +plot_heatmap.square=True \
       +plot_heatmap.cbar=False \
       +plot_heatmap.plot_config_kwargs.font_scale=2 \
       agg_mode=[plot_heatmap] \
       $add_kwargs

python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       "+col_val_subset.beta=[1e-3]" \
       +kwargs.pretty_renamer.Cntr_Stds_Mlpxs="MLP" \
       +kwargs.pretty_renamer.Cntr_Stda="Linear" \
       +kwargs.pretty_renamer.Mlp_H1024_L1="MLP" \
       +plot_heatmap.x="repr" \
       +plot_heatmap.y="pred" \
       +plot_heatmap.cols_to_agg=["seed"] \
       +plot_heatmap.metric="test/pred/acc_mean" \
       +plot_heatmap.filename="heatmap_V" \
       agg_mode=[plot_heatmap]

python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       "+col_val_subset.beta=[1e-3]" \
       +kwargs.pretty_renamer.Cntr_Stda_Mlpxs="MLP" \
       +kwargs.pretty_renamer.Cntr_Stda="Linear" \
       +kwargs.pretty_renamer.Mlp_H1024_L1="MLP" \
       +plot_heatmap.x="repr" \
       +plot_heatmap.y="pred" \
       +plot_heatmap.cols_to_agg=["seed"] \
       +plot_heatmap.metric="test/pred/acc_agg_min_mean" \
       +plot_heatmap.filename="heatmap_V_min" \
       agg_mode=[plot_heatmap]

python utils/aggregate.py \
       experiment=$experiment  \
       agg_mode=[summarize_metrics]