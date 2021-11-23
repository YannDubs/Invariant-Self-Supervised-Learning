#!/usr/bin/env bash

experiment=$prfx"whitening_dev"
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
data_pred.all_data=[data_repr]
predictor=sk_logistic
++data_repr.kwargs.val_size=2
++data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
representor=std_cntr
scheduler@scheduler_issl=whitening
decodability.kwargs.temperature=0.5
data_repr.kwargs.batch_size=512
optimizer_issl.kwargs.weight_decay=1e-6
scheduler_issl.kwargs.base.is_warmup_lr=True
decodability.kwargs.projector_kwargs.hid_dim=1024
decodability.kwargs.projector_kwargs.n_hid_layers=1
encoder.z_shape=512
trainer.max_epochs=1000
data@data_repr=cifar10
decodability.kwargs.projector_kwargs.out_shape=128
timeout=$time
"


# every arguments that you are sweeping over
kwargs_multi="
optimizer_issl.kwargs.lr=3e-3
decodability.kwargs.projector_kwargs.out_shape=64
encoder.is_relu_Z=True
decodability.kwargs.is_bn_proj=True
decodability.kwargs.is_train_temperature=False
"


# difference for gen: linear resnet / augmentations / larger dim




if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "data@data_repr=cifar10_dev" #"data_repr.kwargs.dataset_kwargs.a_augmentations=['std_simclr']" #"scheduler@scheduler_issl=warm_unifmultistep1000" "encoder.is_relu_Z=False decodability.kwargs.is_bn_proj=False" #"" "decodability.kwargs.is_train_temperature=True" "encoder.is_relu_Z=False" "decodability.kwargs.is_bn_proj=False" "decodability.kwargs.src_tgt_comparison=symmetric"
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
