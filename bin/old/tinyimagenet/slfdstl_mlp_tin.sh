#!/usr/bin/env bash

experiment="slfdst_hopt_tin_final"
notes="
**Goal**: hyperparameter tuning for selfdistillation on tinyimagenet.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh



time=10080

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
representor=slfdstl_mlp
data_repr.kwargs.batch_size=256
representor.loss.beta=3e-6
decodability.kwargs.beta_pM_unif=1.7
trainer.max_epochs=500
regularizer=huber
decodability.kwargs.projector_kwargs.bottleneck_size=100
decodability.kwargs.out_dim=50000
decodability.kwargs.ema_weight_prior=0.8
timeout=$time
"

kwargs_multi="
seed=1,2,3
"
# RAN WITH RELU :(

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs  -m &

    sleep 10

  done
fi

wait

# for representor
python utils/aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       agg_mode=[summarize_metrics]
