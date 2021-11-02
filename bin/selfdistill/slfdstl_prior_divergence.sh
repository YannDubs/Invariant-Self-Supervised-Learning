#!/usr/bin/env bash

experiment=$prfx"slfdstl_prior_divergence"
notes="
**Goal**: Hyperparameter tuning of the divergence p(M) and unif for self distillation ISSL compared to standard for linear classification.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
trainer.max_epochs=50
checkpoint@checkpoint_repr=bestTrainLoss
architecture@encoder=resnet18
architecture@online_evaluator=linear
data@data_repr=mnist
data_pred.all_data=[data_repr_agg,data_repr_30,data_repr_100,data_repr_1000]
predictor=sk_logistic
optimizer@optimizer_issl=Adam_lr3e-4_w0
decodability.kwargs.ema_weight_prior=0.9
timeout=$time
$add_kwargs
"


# every arguments that you are sweeping over
kwargs_multi="
representor=slfdstl_prior
seed=1
"
# KL forward seems to work a little better. Why ?


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "decodability.kwargs.beta_pM_unif=null,0,0.1,1,10 decodability.kwargs.divergence=kl_symmetric" "decodability.kwargs.divergence=kl_symmetric,kl_reverse,kl_forward,entropy,tv"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3

  done
fi
