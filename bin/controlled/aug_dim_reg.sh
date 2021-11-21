#!/usr/bin/env bash

experiment=$prfx"aug_dim_reg"
notes="
**Goal**: figure showing effect of augmentations on the necessary dimensionality.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
+logger.wandb_kwargs.project=controlled
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
representor=cntr,cntr_250A,cntr_stdA,cntr_1000A_shuffle,cntr_noA,cntr_coarserA
encoder.z_shape=2,4,8,16,64,250,1024,4096
seed=1,2,3
regularizer=huber
representor.loss.beta=1e-3,1e-1,1e0,1e1
"

kwargs_multi="
representor=cntr,cntr_1000A,cntr_1000A_shuffle,cntr_stdA,cntr_noA,cntr_coarserA
encoder.z_shape=2,8,16,64,250,1024
regularizer=huber
representor.loss.beta=1e0,1e1
seed=1
"

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
       "+col_val_subset.repr=[cntr,cntr_250A,cntr_1000A,cntr_stdA,cntr_noA,cntr_coarserA,cntr_1000A_shuffle]" \
       patterns.representor=null \
       +kwargs.pretty_renamer.Cntr_250A="Minimal" \
       +kwargs.pretty_renamer.Cntr_1000A_Shuffle="Not Sufficient" \
       +kwargs.pretty_renamer.Cntr_1000A="Minimal --" \
       +kwargs.pretty_renamer.Cntr_Stda="Standard" \
       +kwargs.pretty_renamer.Cntr_Noa="None" \
       +kwargs.pretty_renamer.Cntr_Coarsera="Coarser" \
       +kwargs.pretty_renamer.Cntr="Minimal ++" \
       +plot_scatter_lines.x="zdim" \
       +plot_scatter_lines.y="test/pred/accuracy_score_agg_min" \
       +plot_scatter_lines.filename="lines_acc_vs_samples" \
       +plot_scatter_lines.hue="repr" \
       +plot_scatter_lines.style="beta" \
       +plot_scatter_lines.logbase_x=2 \
       +plot_scatter_lines.legend_out=True \
       agg_mode=[plot_scatter_lines]
