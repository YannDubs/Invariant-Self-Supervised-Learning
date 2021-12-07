#!/usr/bin/env bash

experiment=$prfx"cntr_hinge"
notes="
**Goal**: hyperparameter tuning for contrastive.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
+logger.wandb_kwargs.project=cifar10
checkpoint@checkpoint_repr=bestTrainLoss
architecture@encoder=resnet18
architecture@online_evaluator=linear
downstream_task.all_tasks=[sklogistic_datarepr,skknn_datarepr,skknnweighted_datarepr,skrbfsvm_datarepr,skapproxrbfsvm_datarepr]
++data_repr.kwargs.val_size=2
++data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
data_repr.kwargs.batch_size=512
optimizer_issl.kwargs.weight_decay=1e-6
scheduler_issl.kwargs.base.is_warmup_lr=True
trainer.max_epochs=200
data@data_repr=cifar10
optimizer_issl.kwargs.lr=3e-3
scheduler@scheduler_issl=whitening_quick
representor=cntr_hinge_stdA
encoder.z_shape=128
timeout=$time
"


# every arguments that you are sweeping over
kwargs_multi=""


kwargs_multi="
trainer.max_epochs=1
+trainer.limit_train_batches=0.05
+trainer.limit_test_batches=0.05
+data_pred.kwargs.subset_train_size=100
experiment=dev_$experiment
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "" # "encoder.z_shape=2,10,32,128,512,1024" "decodability.kwargs.is_normalize_proj=False"  "decodability.kwargs.is_project=True" "decodability.kwargs.is_hinge=False" "decodability.kwargs.is_kernel=False" "decodability.kwargs.kernel_kwargs.is_normalize=True" "decodability.kwargs.kernel_kwargs.pre_gamma_init=-10,-1,0,1,10" "decodability.kwargs.kernel_kwargs.is_linear=False"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs #-m &

    sleep 10

  done
fi

wait

# for representor
python utils/aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       agg_mode=[summarize_metrics]
