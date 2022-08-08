#!/usr/bin/env bash

experiment=$prfx"dissl_clustering_bs"
notes="
**Goal**: tests whether dissl can recover Mx.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
++logger.wandb_kwargs.project=dissl_controlled
architecture@encoder=resnet18
architecture@online_evaluator=linear
data_repr.kwargs.batch_size=2048
encoder.kwargs.arch_kwargs.is_no_linear=True
data@data_repr=cifar10
data_repr.kwargs.is_force_all_train=True
data_repr.kwargs.is_val_on_test=True
checkpoint@checkpoint_repr=last
optimizer@optimizer_issl=Adam
scheduler@scheduler_issl=cosine
optimizer_issl.kwargs.weight_decay=1e-6
optimizer_issl.kwargs.lr=2e-3
update_trainer_repr.max_epochs=100
regularizer=none
seed=1
++decodability.kwargs.projector_kwargs.n_hid_layers=1
++decodability.kwargs.projector_kwargs.hid_dim=1024
encoder.z_shape=512
is_skip_pred=True
representor=dstl_optA
callbacks.DISSLTeacherClf.is_use=True
decodability.kwargs.out_dim=10
"


kwargs_multi="
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

  done
fi
