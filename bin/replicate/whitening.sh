#!/usr/bin/env bash

experiment=$prfx"whitening"
notes="
**Goal**: ensure that you can replicate results from the whitening paper for stl10, cifar, tinyimagenet with standard contrastive learning.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
+logger.wandb_kwargs.project=replicate
checkpoint@checkpoint_repr=bestTrainLoss
architecture@encoder=resnet18
architecture@online_evaluator=linear
data_pred.all_data=[data_repr_agg,data_repr_p10_test]
predictor=sk_logistic
++data_repr.kwargs.val_size=2
++data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
timeout=$time

"


# every arguments that you are sweeping over
kwargs_multi="
representor=std_cntr
representor=std_cntr,std_cntr_t05
scheduler@scheduler_issl=unifmultistep100,whitening
data_repr.kwargs.batch_size=256
optimizer_issl.kwargs.weight_decay=1e-6
decodability.kwargs.projector_kwargs.hid_dim=1024
decodability.kwargs.projector_kwargs.n_hid_layers=1
scheduler_issl.kwargs.base.is_warmup_lr=True
"

# difference for gen: linear resnet / augmentations / larger dim


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "data@data_repr=cifar10 optimizer_issl.kwargs.lr=1e-3 trainer.max_epochs=1000 decodability.kwargs.projector_kwargs.out_shape=128"  "data@data_repr=stl10_unlabeled optimizer_issl.kwargs.lr=2e-3 trainer.max_epochs=2000 decodability.kwargs.projector_kwargs.out_shape=256" "data@data_repr=tinyimagenet optimizer_issl.kwargs.lr=2e-3 trainer.max_epochs=1000 decodability.kwargs.projector_kwargs.out_shape=256"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 10

  done
fi

wait

# for representor
python utils/aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       agg_mode=[summarize_metrics]


python utils/aggregate.py \
       experiment=$experiment  \
       +col_val_subset.datapred=["stl10_agg"] \
       kwargs.prfx="agg_" \
       agg_mode=[summarize_metrics]