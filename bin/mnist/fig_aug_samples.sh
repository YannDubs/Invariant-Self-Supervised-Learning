#!/usr/bin/env bash

experiment=$prfx"fig_aug_samples"
notes="
**Goal**: figure showing effect of augmentations on the necessary # samples.
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
data_pred.all_data=[data_repr_10,data_repr_20,data_repr_30,data_repr_50,data_repr_100,data_repr_300,data_repr_1000,data_repr_3000,data_repr_10000,data_repr]
predictor=sk_logistic
data_repr.kwargs.val_size=2
+data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
timeout=$time

"


# every arguments that you are sweeping over
kwargs_multi="
representor=cntr,cntr_250A,cntr_1000A,cntr_stdA,cntr_noA,cntr_coarserA
seed=1,2,3
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
       "+col_val_subset.repr=[cntr,cntr_250A,cntr_1000A,cntr_stdA,cntr_noA,cntr_coarserA]" \
       patterns.representor=null \
       +kwargs.pretty_renamer.Cntr_60A="Minimal" \
       +kwargs.pretty_renamer.Cntr_250A="Minimal--" \
       +kwargs.pretty_renamer.Cntr_Stda="Standard" \
       +kwargs.pretty_renamer.Cntr_Noa="None" \
       +kwargs.pretty_renamer.Cntr_Coarsera="Not Sufficient" \
       +kwargs.pretty_renamer.Cntr="Minimal++" \
       +collect_data.params_to_add.n_samples="data.kwargs.subset_train_size" \
       +plot_scatter_lines.x="n_samples" \
       +plot_scatter_lines.y="test/pred/accuracy_score" \
       +plot_scatter_lines.filename="lines_acc_vs_samples" \
       +plot_scatter_lines.hue="repr" \
       +plot_scatter_lines.style="repr" \
       +plot_scatter_lines.logbase_x=null \
       +plot_scatter_lines.legend_out=False \
       agg_mode=[plot_scatter_lines]


python utils/aggregate.py \
       experiment=$experiment  \
       agg_mode=[summarize_metrics]