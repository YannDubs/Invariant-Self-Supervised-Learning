#!/usr/bin/env bash

experiment=$prfx"cntrlld_V_heatmap_final"
notes="
**Goal**: figure showing effect of predictive family depending on downstream family.
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
predictor=pytorch
data_repr.kwargs.val_size=2
+data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
representor=exact
timeout=$time
"


# every arguments that you are sweeping over
kwargs_multi="
architecture@predictor=linear,mlp_h10_l1,mlp_h2048_l2
representor=exact,exact_mlpnano,exact_mlp
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


# add split between yours and standard
# add renaming
python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       +kwargs.pretty_renamer.Exact_Mlpnano="MLP --" \
       +kwargs.pretty_renamer.Exact_Mlp="MLP ++" \
       +kwargs.pretty_renamer.Exact="Linear" \
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
       +kwargs.pretty_renamer.Exact_Mlpnano="MLP --" \
       +kwargs.pretty_renamer.Exact_Mlp="MLP ++" \
       +kwargs.pretty_renamer.Exact="Linear" \
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