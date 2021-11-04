#!/usr/bin/env bash

experiment=$prfx"approx_subtasks"
notes="
**Goal**: evaluate the approximation on the number of subtasks
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
data_pred.all_data=[data_repr_agg4,data_repr_agg8,data_repr_agg10,data_repr_agg16,data_repr_agg32,data_repr_agg64,data_repr_agg128,data_repr_agg4_mult,data_repr_agg8_mult,data_repr_agg10_mult,data_repr_agg16_mult,data_repr_agg32_mult,data_repr_agg64_mult,data_repr_agg128_mult]
predictor=sk_logistic
data_repr.kwargs.val_size=2
+data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
timeout=$time
$add_kwargs
"


# every arguments that you are sweeping over
kwargs_multi="
representor=std_gen_smallZ,cntr_stdA
seed=1,2,3
"


# difference for gen: linear resnet / augmentations / larger dim


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3

  done
fi

wait 