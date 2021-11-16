#!/usr/bin/env bash

experiment=$prfx"understand_coarserA"
notes="
**Goal**: understand and debug the effect of finegraining / coarsegraining the equivalence class.
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
data_pred.all_data=[data_repr_agg,data_repr_10,data_repr_100,data_repr_1000]
predictor=sk_logistic
data_repr.kwargs.val_size=2
+data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
optimizer@optimizer_issl=Adam_lr1e-4_w0
representor=cntr
timeout=$time
$add_kwargs
"


#representor=cntr,cntr_1000A,cntr_100000A,cntr_noA,cntr_coarserA
kwargs_multi="
data@data_repr=mnist,mnist_shuffled
representor=exact,exact_1000A,exact_1000A_shuffle,exact_50000A,exact_noA,exact_coarserA
regularizer=l2Mx
representor.loss.beta=1e-1
+data_repr.kwargs.is_shuffle_train=False
"


# difference for gen: linear resnet / augmentations / larger dim


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do
    # on mnist typically z_shape would be quite small but we say that it should be larger

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3

  done
fi

wait

