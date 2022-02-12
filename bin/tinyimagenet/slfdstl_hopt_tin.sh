#!/usr/bin/env bash

experiment="slfdst_hopt_tin"
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
representor=slfdstl
data_repr.kwargs.batch_size=256
representor.loss.beta=1e-6
decodability.kwargs.beta_pM_unif=1.7
decodability.kwargs.ema_weight_prior=null
trainer.max_epochs=1000
decodability.kwargs.out_dim=10000
regularizer=huber
timeout=$time
"

kwargs_multi="
seed=3
decodability.kwargs.temperature=1,0.1
"

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
