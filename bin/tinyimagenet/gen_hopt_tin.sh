#!/usr/bin/env bash

experiment="gen_hopt_tin_final"
notes="
**Goal**: hyperparameter tuning for generative on tinyimagenet.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10080

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
representor=gen
data_repr.kwargs.batch_size=256
optimizer_issl.kwargs.weight_decay=2e-6
representor.loss.beta=2e-6
regularizer=huber
decodability.kwargs.predecoder_kwargs.bottleneck_size=100
decodability.kwargs.predecode_n_Mx=30000
timeout=$time
"

# every arguments that you are sweeping over
kwargs_multi="
seed=1
trainer.max_epochs=300
"
#,2,3
# /juice/scr/yanndubs/Invariant-Self-Supervised-Learning/outputs/2022-02-21_15-26-53

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
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
