#!/usr/bin/env bash

experiment=$prfx"cntrlld_approx_subtasks"
notes="
**Goal**: evaluate the approximation on the number of subtasks
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
data_pred.all_data=[data_repr_agg4,data_repr_agg10,data_repr_agg16,data_repr_agg32,data_repr_agg64,data_repr_agg4_mult,data_repr_agg10_mult,data_repr_agg16_mult,data_repr_agg32_mult,data_repr_agg64_mult]
predictor=sk_logistic
data_repr.kwargs.val_size=2
+data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
timeout=$time
"


# every arguments that you are sweeping over
kwargs_multi="
representor=exact,exact_stdA
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


# for predictor
python utils/aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       patterns.representor=null \
       +cols_to_str=["tasks"] \
       +fillna.tasks="All" \
       +sort="repr" \
       "+col_val_subset.n_tasks=[4,10,16,32,64]" \
       "+col_val_subset.repr=[exact,exact_stdA]" \
       +kwargs.pretty_renamer.Exact="Exact" \
       +kwargs.pretty_renamer.Exact_stda="Exact std. aug." \
       +collect_data.params_to_add.n_tasks="data.kwargs.dataset_kwargs.n_agg_tasks" \
       +collect_data.params_to_add.tasks="data.kwargs.dataset_kwargs.max_k_ary_agg" \
       +plot_scatter_lines.x="n_tasks" \
       +plot_scatter_lines.y="train/pred/accuracy_score_agg_min" \
       +plot_scatter_lines.filename="lines_acc_vs_agg" \
       +plot_scatter_lines.hue="tasks" \
       +plot_scatter_lines.style="tasks" \
       +plot_scatter_lines.col="repr" \
       +plot_scatter_lines.logbase_x=2 \
       agg_mode=[plot_scatter_lines]

python utils/aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       agg_mode=[summarize_metrics]