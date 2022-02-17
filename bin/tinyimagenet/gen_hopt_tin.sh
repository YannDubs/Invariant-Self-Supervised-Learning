#!/usr/bin/env bash

experiment="gen_hopt_tin" # should add final
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
timeout=$time
"

# every arguments that you are sweeping over
kwargs_multi="
seed=1,2,3
trainer.max_epochs=1000
"


kwargs_multi="
seed=3
trainer.max_epochs=1000
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "decodability.kwargs.predecode_n_Mx=3000 decodability.kwargs.predecoder_kwargs.bottleneck_size=null" "decodability.kwargs.predecode_n_Mx=10000 decodability.kwargs.predecoder_kwargs.bottleneck_size=100" "decodability.kwargs.predecode_n_Mx=30000 decodability.kwargs.predecoder_kwargs.bottleneck_size=50" "decodability.kwargs.predecode_n_Mx=90000 decodability.kwargs.predecoder_kwargs.bottleneck_size=30"
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
