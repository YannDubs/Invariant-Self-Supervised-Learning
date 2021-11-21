#!/usr/bin/env bash

experiment=$prfx"fig_aug_dim_exact"
notes="
**Goal**: figure showing effect of augmentations on the necessary dimensionality.
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
encoder.z_shape=2,4,8,16,32,64,128,250,512,1024,2048
seed=1
"
# TODO: run seed 2,3

# difference for gen: linear resnet / augmentations / larger dim


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
       +kwargs.pretty_renamer.Exact_Stda="Standard" \
       +kwargs.pretty_renamer.Exact_Noa="None" \
       +kwargs.pretty_renamer.Exact_Coarsera="Not Sufficient" \
       +kwargs.pretty_renamer.Exact="Minimal++" \
       +plot_scatter_lines.x="zdim" \
       +plot_scatter_lines.y="test/pred/accuracy_score" \
       +plot_scatter_lines.filename="lines_acc_vs_samples" \
       +plot_scatter_lines.hue="repr" \
       +plot_scatter_lines.style="repr" \
       +plot_scatter_lines.logbase_x=2 \
       +plot_scatter_lines.legend_out=False \
       agg_mode=[plot_scatter_lines]