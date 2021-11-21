#!/usr/bin/env bash

experiment=$prfx"debug_shuffled"
notes="
**Goal**: debug mnist shuffled to make sure it works.
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
data_pred.all_data=[data_repr_300]
predictor=sk_logistic
optimizer@optimizer_issl=Adam_lr1e-4_w0
scheduler@scheduler_issl=unifmultistep1000
data_repr.kwargs.val_size=2
+data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
timeout=$time

"

kwargs_multi="
data@data_repr=mnist_shuffled
representor=exact
regularizer=l2Mx
"

# difference for gen: linear resnet / augmentations / larger dim


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "" #"representor=exact regularizer=l2Mx" "representor=cntr regularizer=huber"
  do
    # on mnist typically z_shape would be quite small but we say that it should be larger

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 3

  done
fi

wait

