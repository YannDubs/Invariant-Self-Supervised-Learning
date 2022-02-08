#!/usr/bin/env bash

experiment="slfdst_hopt_ema_tin"
notes="
**Goal**: make sefldistillation work with ema.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh



time=4000

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
representor=slfdstl_augIN
data_repr.kwargs.batch_size=256
decodability.kwargs.ema_weight_prior=0.5
scheduler_issl.kwargs.UniformMultiStepLR.k_steps=5
scheduler_issl.kwargs.UniformMultiStepLR.decay_per_step=3
timeout=$time
"

kwargs_multi="
seed=3
trainer.max_epochs=500
decodability.kwargs.out_dim=500,7000
representor.loss.beta=1e-6
decodability.kwargs.beta_pM_unif=1.7
regularizer=huber
optimizer_issl.kwargs.weight_decay=1e-6
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
