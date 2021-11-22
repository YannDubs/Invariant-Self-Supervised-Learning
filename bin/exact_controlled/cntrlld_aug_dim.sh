#!/usr/bin/env bash

experiment=$prfx"cntrlld_aug_dim"
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
representor=exact,exact_1000A,exact_1000A_shuffle,exact_stdA,exact_noA
encoder.z_shape=5,10,100,1000,10000
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
       patterns.representor=null \
       +kwargs.pretty_renamer.Exact_250A="Minimal" \
       +kwargs.pretty_renamer.Exact_1000A="Minimal--" \
       +kwargs.pretty_renamer.Exact_1000A_Shuffle="Minimal--" \
       +kwargs.pretty_renamer.Exact_Stda="Standard" \
       +kwargs.pretty_renamer.Exact_Noa="None" \
       +kwargs.pretty_renamer.Exact_Coarsera="Coarser" \
       +kwargs.pretty_renamer.Exact="Minimal++" \
       +plot_scatter_lines.x="zdim" \
       +plot_scatter_lines.y="train/pred/accuracy_score_agg_min" \
       +plot_scatter_lines.filename="lines_acc_vs_samples" \
       +plot_scatter_lines.hue="repr" \
       +plot_scatter_lines.style="repr" \
       +plot_scatter_lines.logbase_x=10 \
       +plot_scatter_lines.legend_out=False \
       agg_mode=[plot_scatter_lines]