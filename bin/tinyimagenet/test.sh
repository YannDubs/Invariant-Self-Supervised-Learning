#!/usr/bin/env bash

experiment="test"
notes="
**Goal**: hyperparameter tuning for contrastive on tinyimagenet.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10080

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
++logger.wandb_kwargs.project=dev
representor=cntr
++trainer.limit_predict_batches=10
++trainer.limit_train_batches=10
++trainer.limit_test_batches=10
++trainer.limit_test_batches=10
data@data_repr=mnist
decodability.kwargs.temperature=0.07
downstream_task.all_tasks=[pytorch_datarepr01test,pytorch_datarepr,pytorch_bn_datarepr]
timeout=$time
"

# every arguments that you are sweeping over
kwargs_multi="
seed=3
trainer.max_epochs=2
update_trainer_pred.max_epochs=2
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs

    sleep 10

  done
fi

wait

# for representor
python utils/aggregate.py \
       experiment=$experiment  \
       agg_mode=[summarize_metrics]
