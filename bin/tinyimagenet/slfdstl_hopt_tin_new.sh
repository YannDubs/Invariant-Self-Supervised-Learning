#!/usr/bin/env bash

experiment="slfdst_hopt_tin"
notes="
**Goal**: hyperparameter tuning for selfdistillation on tinyimagenet.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh



time=5080

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
representor=slfdstl
data_repr.kwargs.batch_size=256
representor.loss.beta=3e-6
decodability.kwargs.beta_pM_unif=1.7
regularizer=huber
timeout=$time
"

kwargs_multi="
seed=3
trainer.max_epochs=500
decodability.kwargs.out_dim=10000
decodability.kwargs.ema_weight_prior=null,0.5
decodability.kwargs.projector_kwargs.architecture=cosine
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs  -m &

    sleep 10

  done
fi
