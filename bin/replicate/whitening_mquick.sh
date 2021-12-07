#!/usr/bin/env bash

experiment=$prfx"whitening_mquick"
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
downstream_task.all_tasks=[sklogistic_datarepr]
++data_repr.kwargs.val_size=2
++data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
representor=std_cntr
decodability.kwargs.temperature=0.5
data_repr.kwargs.batch_size=512
optimizer_issl.kwargs.weight_decay=1e-6
scheduler_issl.kwargs.base.is_warmup_lr=True
decodability.kwargs.projector_kwargs.hid_dim=1024
decodability.kwargs.projector_kwargs.n_hid_layers=1
encoder.z_shape=512
trainer.max_epochs=300
data@data_repr=cifar10
decodability.kwargs.projector_kwargs.out_shape=128
optimizer_issl.kwargs.lr=2e-3
timeout=$time
"


# every arguments that you are sweeping over
kwargs_multi="
scheduler@scheduler_issl=warm_unifmultistep125,whitening_quick,warm_unifmultistep100,warm_unifmultistep25,cosine
data@data_repr=tinyimagenet trainer.max_epochs=500,1000
"

kwargs_multi="
scheduler@scheduler_issl=cosine
data@data_repr=tinyimagenet trainer.max_epochs=500,1000
"
# use warm_unifmultistep25 with 500 and warm_unifmultistep100 with 1000




if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in   ""
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