#!/usr/bin/env bash

experiment=$prfx"cntr_hopt_tin"
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
representor=cntr
optimizer_issl.kwargs.weight_decay=5e-6
decodability.kwargs.temperature=0.07
timeout=$time
"

# every arguments that you are sweeping over
kwargs_multi="
seed=3
trainer.max_epochs=1000
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "" "scheduler_issl.kwargs.UniformMultiStepLR.decay_per_step=3" "scheduler_issl.kwargs.UniformMultiStepLR.decay_per_step=5 scheduler_issl.kwargs.UniformMultiStepLR.k_steps=3" "scheduler_issl.kwargs.UniformMultiStepLR.decay_per_step=3 scheduler_issl.kwargs.UniformMultiStepLR.k_steps=3"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 10

  done
fi

wait

# for representor
python utils/aggregate.py \
       experiment=$experiment  \
       agg_mode=[summarize_metrics]
