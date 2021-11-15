#!/usr/bin/env bash

experiment=$prfx"fig_V_heatmap_exact"
notes="
**Goal**: figure showing effect of predictive family depending on downstream family.
"

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
representor=std_cntr
timeout=$time
$add_kwargs
"


# every arguments that you are sweeping over
kwargs_multi="
architecture@predictor=linear,mlp_h1024_l1,mlp_h2048_l3
seed=1
"


# need to rerun with seed: 3 once happy


# difference for gen: linear resnet / augmentations / larger dim


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "representor=exact_stdA,exact_stdA_mlpS,exact_stdA_mlpL"
  do
    # on mnist typically z_shape would be quite small but we say that it should be larger

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3

  done
fi

wait


# add split between yours and standard
# add renaming
python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       +plot_heatmap.x="pred" \
       +plot_heatmap.y="repr" \
       +plot_heatmap.cols_to_agg=["seed"] \
       +plot_heatmap.metric="test/pred/acc_agg_mean" \
       +plot_heatmap.filename="heatmap_V" \
       agg_mode=[plot_heatmap]


python utils/aggregate.py \
       experiment=$experiment  \
       agg_mode=[summarize_metrics]